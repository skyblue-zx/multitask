import time
import yaml
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from model_core.src.utils.metric import f1_score

from model_core.src.utils.log import LoggerHelper

from model_core.src.data.didi_dataset_text import DiDiDatasetText


class TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge(nn.Module):
    """
    文本单模态
    无预训练
    """
    def __init__(self, config_file_path):
        """
        构造函数
        :param text_pretrain_model:
        :param text_embedding_dim:
        """
        super(TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge, self).__init__()

        # 读取配置文件
        with open(config_file_path, 'r') as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

        # 设置文本维度
        self.text_dim = config['text_dim']
        # 设置文本嵌入
        if config['embedding_path'] is None:
            self.text_embeddings = nn.Embedding(config['vocabulary_size'] + 1, self.text_dim)
        else:
            text_embeddings = np.load(config['embedding_path'])
            zero_embedding = np.zeros((1, text_embeddings.shape[1]))
            text_embeddings = np.concatenate([zero_embedding, text_embeddings], axis=0)
            self.text_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(text_embeddings), freeze=False)

        # 辅助网络
        self.auxiliary_bilstm = nn.LSTM(input_size=self.text_dim, hidden_size=config['text_hidden'],
                                   num_layers=1, batch_first=True, bidirectional=True)

        self.auxiliary_predict_linear = nn.Linear(config['text_hidden'] * 2, config['class_num'])

        # 设置文本双向LSTM网络
        # 输入维度为文本维度
        self.text_bilstm = nn.LSTM(input_size=self.text_dim, hidden_size=config['text_hidden'],
                                   num_layers=1, batch_first=True, bidirectional=True)

        self.predict_linear = nn.Linear(config['text_hidden'] * 2, config['class_num'])

        # Gate网络
        self.gate = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.text_dim, self.text_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.text_dim, 2, bias=True),
            nn.Softmax(dim=1)
        )
        #self.gate_w = torch.nn.Parameter(torch.rand([2, 1]), requires_grad=True)
        #self.gate_b = torch.nn.Parameter(torch.randn([1, 2]), requires_grad=True)

    def forward(self, text_inputs, text_length):
        """
        前馈函数
        :param asr_tokens:
        :param asr_segment_ids:
        :param asr_attention_mask:
        :return:
        """
        device = text_inputs.device

        # 文本嵌入
        text_embedding = self.text_embeddings(text_inputs)

        # 初始化文本mask
        # 同上
        text_mask = torch.arange(text_inputs.size(1))[None, :].repeat(text_inputs.size(0), 1).to(device)
        text_mask = (text_mask < text_length[:, None].repeat(1, text_inputs.size(1))).float()

        total_length = text_inputs.size(1)

        # 按照文本序列实际长度对文本序列进行Pack
        auxiliary_text_pack = nn.utils.rnn.pack_padded_sequence(text_embedding, text_length, batch_first=True, enforce_sorted=False)
        # 通过双向LSTM对文本编码
        auxiliary_text_encode, _ = self.auxiliary_bilstm(auxiliary_text_pack)
        # 恢复文本编码序列的Pad形式
        auxiliary_text_encode, _ = nn.utils.rnn.pad_packed_sequence(auxiliary_text_encode, batch_first=True, total_length=total_length)

        # 将Pad部分的值修改为-10000
        # 因为非Pad部分可能出现负值，为了在MaxPool操作时Pad部分不参与运算，需要将其值修改为极小值
        auxiliary_text_maxpool = auxiliary_text_encode * text_mask[:, :, None] - 10000.0 * (1 - text_mask[:, :, None])
        # MaxPool操作
        auxiliary_text_maxpool, _ = torch.max(auxiliary_text_maxpool, dim=1)  # [b,dim]

        # 对音频和文本融合后的特征编码进行FC计算
        auxiliary_logits = self.auxiliary_predict_linear(auxiliary_text_maxpool)

        # 按照文本序列实际长度对文本序列进行Pack
        text_pack = nn.utils.rnn.pack_padded_sequence(text_embedding, text_length, batch_first=True, enforce_sorted=False)
        # 通过双向LSTM对文本编码
        text_encode, _ = self.text_bilstm(text_pack)
        # 恢复文本编码序列的Pad形式
        text_encode, _ = nn.utils.rnn.pad_packed_sequence(text_encode, batch_first=True, total_length=total_length)
       
        # 将Pad部分的值修改为-10000
        # 因为非Pad部分可能出现负值，为了在MaxPool操作时Pad部分不参与运算，需要将其值修改为极小值
        text_maxpool = text_encode * text_mask[:, :, None] - 10000.0 * (1 - text_mask[:, :, None])
        # MaxPool操作
        text_maxpool, _ = torch.max(text_maxpool, dim=1)  # [b,dim]

        # 对音频和文本融合后的特征编码进行FC计算
        logits = self.predict_linear(text_maxpool)

        # 合并全部模型的输出
        gate_output = self.gate(torch.max(text_embedding, dim=1)[0])
        gate_output = gate_output.unsqueeze(2)

        merged_output = torch.cat((logits.unsqueeze(1), auxiliary_logits.unsqueeze(1)), 1)
        #gate_output = nn.Softmax(dim=0)(self.gate_w)
        gated_output = merged_output * gate_output
        main_output = gated_output.sum(dim=1)

        return main_output, auxiliary_logits

    @classmethod
    def train_model(cls,
                    master_gpu_id,
                    model,
                    optimizer,
                    scheduler,
                    data_loader,
                    gradient_accumulation_steps,
                    use_cuda):
        """

        :param master_gpu_id:
        :param model:
        :param optimizer:
        :param scheduler:
        :param data_loader:
        :param gradient_accumulation_steps:
        :param use_cuda:
        :return:
        """
        model.train()

        loss_function = nn.CrossEntropyLoss()
        loss_loss_criterion = nn.L1Loss()

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        if hasattr(model, 'module'):
            current_model = model.module
        else:
            current_model = model

        for step, batch in enumerate(data_loader):
            # 记录开始时间
            start_t = time.time()

            # 获取Batch数据
            text_inputs, label_inputs, asr_label_inputs = batch
            # 处理数据放置位置
            # label数据
            label_inputs = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs
            asr_label_inputs = asr_label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else asr_label_inputs
            # 文本字符向量和文本长度
            text_inputs, text_length = text_inputs
            text_inputs = text_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_inputs
            text_length = text_length.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_length

            # 获得模型输出
            prediction, auxiliary_prediction = model(text_inputs, text_length)
            # 计算模型loss
            main_loss = loss_function(prediction, label_inputs)
            auxiliary_loss = loss_function(auxiliary_prediction, asr_label_inputs)

            # 计算加权总损失
            weighted_loss = current_model.loss_weight(main_loss, auxiliary_loss)
            # 求加权损失累积均值
            if gradient_accumulation_steps > 1:
                weighted_loss /= gradient_accumulation_steps

            # 记录第一个Batch的损失
            # 用于之后计算评估训练速度的参数
            if step == 0:
                initial_auxiliary_loss = auxiliary_loss.detach()
                initial_main_loss = main_loss.detach()

            # 判断是否到达累计总步数
            # 若到达则清空累计梯度
            if step % gradient_accumulation_steps == 0:
                model.zero_grad()

            # 损失反向传播且不清除计算图
            weighted_loss.backward(retain_graph=True)
            
            text_embeddings_last_layer_params = list(current_model.text_embeddings.parameters())

            # 计算ASR模型损失到三层FC的梯度
            auxiliary_gradient = torch.autograd.grad(current_model.loss_weight.auxiliary_model_weight * auxiliary_loss,
                                                     text_embeddings_last_layer_params[-1],
                                                     retain_graph=True,
                                                     create_graph=True)
            # 计算上述梯度中对应FC第一层Linear的梯度并加权求二范数
            auxiliary_norms = torch.norm(auxiliary_gradient[0], 2)

            # 计算主模型损失到MobileNetV2的梯度
            main_gradient = torch.autograd.grad(current_model.loss_weight.main_model_weight * main_loss,
                                                text_embeddings_last_layer_params[-1],
                                                retain_graph=True,
                                                create_graph=True)
            # 计算上述梯度中对应MobileNetV2第一层的梯度并加权求二范数
            main_norms = torch.norm(main_gradient[0], 2)

            # 求上述两个梯度二范数的均值
            mean_norm = torch.div(torch.add(auxiliary_norms, main_norms), 2)

            # 计算ASR模型当前Batch loss与首个Batch loss的比例
            auxiliary_loss_ratio = torch.div(auxiliary_norms, initial_auxiliary_loss)
            # 计算主模型当前Batch loss与首个Batch loss的比例
            main_loss_ratio = torch.div(main_loss.data, initial_main_loss)
            mean_loss_ratio = torch.div(torch.add(auxiliary_loss_ratio, main_loss_ratio), 2)

            # 计算ASR模型当前的训练速度参数
            auxiliary_train_rate = torch.div(auxiliary_loss_ratio, mean_loss_ratio)
            # 计算主模型当前的训练速度参数
            main_train_rate = torch.div(main_loss_ratio, mean_loss_ratio)

            # Todo
            # 超参读入
            auxiliary_loss_target = mean_norm * (auxiliary_train_rate) ** 0.16
            main_loss_target = mean_norm * (main_train_rate) ** 0.16
            auxiliary_loss_target = auxiliary_loss_target.detach()
            main_loss_target = main_loss_target.detach()

            optimizer[1].zero_grad()
            loss_sum = torch.add(loss_loss_criterion(auxiliary_norms, auxiliary_loss_target),
                                 loss_loss_criterion(main_norms, main_loss_target))
            loss_sum.backward()

            optimizer[1].step()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer[0].step()
                #scheduler.step()

            loss_value = weighted_loss.item()
            _, top_index = prediction.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            correct_sum += (top_index.view(-1) == label_inputs).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info(
                "step: {}\tloss: {:.2f}\tmain_loss: {:.2f}\tasr_loss: {:.2f}\ttime(s): {:.2f}".format(step, loss_value,
                                                                                                      main_loss,
                                                                                                      auxiliary_loss, cost_t))

            normalize_coef = 2 / torch.add(current_model.loss_weight.auxiliary_model_weight.data,
                                           current_model.loss_weight.main_model_weight.data)
            current_model.loss_weight.auxiliary_model_weight.data = current_model.loss_weight.auxiliary_model_weight.data * normalize_coef
            current_model.loss_weight.main_model_weight.data = current_model.loss_weight.main_model_weight.data * normalize_coef

            LoggerHelper.info(
                "asr loss weight: {}\tmain loss weight: {}".format(model.module.loss_weight.auxiliary_model_weight.item(),
                                                                   model.module.loss_weight.main_model_weight.item()))

        LoggerHelper.info("Total Training Samples: " + str(num_sample))
        LoggerHelper.info("Correct Prediction: " + str(correct_sum))
        LoggerHelper.info("Error Rate: " + format(1 - (correct_sum / num_sample), "0.4f"))

        return total_loss / num_batch

    @classmethod
    def eval_model(cls,
                   master_gpu_id,
                   model,
                   eval_dataset,
                   eval_batch_size=1,
                   use_cuda=False,
                   num_workers=1):
        model.eval()

        eval_dataloader = DataLoader(dataset=eval_dataset,
                                     pin_memory=use_cuda,
                                     batch_size=eval_batch_size,
                                     num_workers=num_workers,
                                     shuffle=False,
                                     collate_fn=DiDiDatasetText.collate)

        predicted_probs = []
        true_labels = []

        batch_count = 1
        for batch in tqdm(eval_dataloader, unit="batch", ncols=100, desc="Evaluating process: "):
            text_inputs, label_inputs, _ = batch

            label_inputs = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs

            text_inputs, text_length = text_inputs
            text_inputs = text_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_inputs
            text_length = text_length.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_length

            with torch.no_grad():
                main_output, _ = model(text_inputs, text_length)

                # 将模型输出转为列表
                main_output = torch.softmax(main_output, dim=1).cpu().tolist()
                # 获取正例结果
                prob = np.array(main_output)[:, 1]
                # 将该Batch的正例预测值列表拼接至全局正例预测值列表中
                predicted_probs.extend(prob.tolist())

                # 将真实label列表拼接至全局真实label列表
                true_labels.extend(label_inputs.tolist())

                LoggerHelper.info("Batch: " + str(batch_count))
                batch_count += 1

        predicted_probs = [round(prob, 2) for prob in predicted_probs]
        precision, recall, _thresholds = precision_recall_curve(true_labels, predicted_probs)
        auc = roc_auc_score(true_labels, predicted_probs)
        logloss = log_loss(true_labels, predicted_probs)
        for i in range(len(_thresholds)):
            log_str_th = 'VAL => Thresholds: {0:>2}, Precision: {1:>7.2%}, Recall: {2:>7.2%}, F1: {3:>7.2%}'.format(
                _thresholds[i], precision[i], recall[i], f1_score(precision[i], recall[i]))
            LoggerHelper.info(log_str_th)

        LoggerHelper.info("AUC: " + str(auc))
        LoggerHelper.info("Logloss: " + str(logloss))

        return
