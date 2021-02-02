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

from model_core.src.models.text_bert_embedding_model import TextBertEmbeddingModel


class AudioSingleModelBasedOnLSTMTextGateMerge(nn.Module):
    def __init__(self, config_file_path, asr_pretrain_model, asr_embedding_dim, audio_embedding_dim):
        super(AudioSingleModelBasedOnLSTMTextGateMerge, self).__init__()

        # 读取配置文件
        with open(config_file_path, 'r') as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

        # 设置文本嵌入模型
        self.asr_embedding_model = TextBertEmbeddingModel(asr_pretrain_model)
        # 获取文本嵌入模型的嵌入维度
        self.asr_embedding_model_dim = self.asr_embedding_model.text_embedding_dim
        # 设置文本嵌入维度
        self.asr_embedding_dim = asr_embedding_dim

        # 设置音频嵌入维度
        self.audio_embedding_dim = audio_embedding_dim

        # ASR二分类
        self.asr_model = nn.Sequential(
            nn.Linear(self.asr_embedding_model_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, 2)
        )

        # 设置音频双向LSTM网络
        # 输入维度为音频维度
        self.audio_bilstm = nn.LSTM(input_size=self.audio_embedding_dim, hidden_size=config['audio_hidden'],
                                    num_layers=1, batch_first=True, bidirectional=True)

        self.predict_linear = nn.Linear(config['audio_hidden'] * 2, config['class_num'])

        self.gate_w = torch.nn.Parameter(torch.rand([2, 1]), requires_grad=True)
        self.gate_b = torch.nn.Parameter(torch.randn([1, 2]), requires_grad=True)

    def forward(self, asr_tokens, asr_segment_ids, asr_attention_mask, audio_inputs, audio_length):
        device = audio_inputs.device

        # 文本子模型
        asr_embedding = self.asr_embedding_model(asr_tokens, asr_segment_ids, asr_attention_mask)

        # ASR文本子模型输出
        asr_output = self.asr_model(asr_embedding)

        # 按照音频序列实际长度对音频序列进行Pack
        audio_pack = nn.utils.rnn.pack_padded_sequence(audio_inputs, audio_length, batch_first=True,
                                                       enforce_sorted=False)
        # 使用双向LSTM对音频编码
        audio_encode, _ = self.audio_bilstm(audio_pack)
        # 恢复音频编码序列的Pad形式
        total_length = audio_inputs.size(1)
        audio_encode, _ = nn.utils.rnn.pad_packed_sequence(audio_encode, batch_first=True, total_length=total_length)

        # 初始化音频mask
        # mask为二维矩阵，每行对应一条样本，为从1到最长样本长度的序列
        audio_mask = torch.arange(audio_inputs.size(1))[None, :].repeat(audio_inputs.size(0), 1).to(device)
        # 根据实际长度生成元素为0或1的二维矩阵，每行对应一条样本，其中1表示有效元素，0表示Pad造成的无效0元素
        audio_mask = (audio_mask < audio_length[:, None].repeat(1, audio_inputs.size(1))).float()

        # 将Pad部分的值修改为-10000
        # 因为非Pad部分可能出现负值，为了在MaxPool操作时Pad部分不参与运算，需要将其值修改为极小值
        audio_maxpool = audio_encode * audio_mask[:, :, None] - 10000.0 * (1 - audio_mask[:, :, None])
        # MaxPool操作
        audio_maxpool, _ = torch.max(audio_maxpool, dim=1)  # [b,dim]

        # 对音频和文本融合后的特征编码进行FC计算
        main_output = self.predict_linear(audio_maxpool)

        merged_output = torch.cat((main_output.unsqueeze(1), asr_output.unsqueeze(1)), 1)
        gate_output = nn.Softmax(dim=0)(self.gate_w)
        gated_output = merged_output * gate_output
        main_output = gated_output.sum(dim=1)

        return main_output, asr_output

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

        loss_criterion = nn.CrossEntropyLoss()
        loss_loss_criterion = nn.L1Loss()

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
        for step, batch in enumerate(data_loader):
            start_t = time.time()

            # 获取label和音频数据并配置GPU
            label_inputs = batch['label'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['label']
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                'segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]

            audio_inputs = batch['audio'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio']
            audio_length = batch['audio_length'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio_length']

            main_output, asr_output = model(tokens, segment_ids, attention_mask, audio_inputs, audio_length)

            # 计算主模型损失和ASR模型损失
            main_loss = loss_criterion(main_output, label_inputs)
            asr_loss = loss_criterion(asr_output, asr_label)

            # 计算加权总损失
            weighted_loss = model.module.loss_weight(main_loss, asr_loss)
            # 求加权损失累积均值
            if gradient_accumulation_steps > 1:
                weighted_loss /= gradient_accumulation_steps

            # 记录第一个Batch的损失
            # 用于之后计算评估训练速度的参数
            if step == 0:
                initial_asr_loss = asr_loss.detach()
                initial_main_loss = main_loss.detach()

            # 判断是否到达累计总步数
            # 若到达则清空累计梯度
            if step % gradient_accumulation_steps == 0:
                model.zero_grad()

            # 损失反向传播且不清除计算图
            weighted_loss.backward(retain_graph=True)

            # 计算ASR模型损失到三层FC的梯度

            asr_gradient = torch.autograd.grad(model.module.loss_weight.asr_model_weight * asr_loss,
                                               model.module.asr_model.parameters(),
                                               retain_graph=True,
                                               create_graph=True)
            # 计算上述梯度中对应FC第一层Linear的梯度并加权求二范数
            asr_norms = torch.norm(asr_gradient[0], 2)

            # 计算主模型损失到MobileNetV2的梯度
            main_gradient = torch.autograd.grad(model.module.loss_weight.main_model_weight * main_loss,
                                                model.module.main_model.parameters(),
                                                retain_graph=True,
                                                create_graph=True)
            # 计算上述梯度中对应MobileNetV2第一层的梯度并加权求二范数
            main_norms = torch.norm(main_gradient[0], 2)

            # 求上述两个梯度二范数的均值
            mean_norm = torch.div(torch.add(asr_norms, main_norms), 2)

            # 计算ASR模型当前Batch loss与首个Batch loss的比例
            asr_loss_ratio = torch.div(asr_loss, initial_asr_loss)
            # 计算主模型当前Batch loss与首个Batch loss的比例
            main_loss_ratio = torch.div(main_loss.data, initial_main_loss)
            mean_loss_ratio = torch.div(torch.add(asr_loss_ratio, main_loss_ratio), 2)

            # 计算ASR模型当前的训练速度参数
            asr_train_rate = torch.div(asr_loss_ratio, mean_loss_ratio)
            # 计算主模型当前的训练速度参数
            main_train_rate = torch.div(main_loss_ratio, mean_loss_ratio)

            # Todo
            # 超参读入
            asr_loss_target = mean_norm * (asr_train_rate) ** 0.16
            main_loss_target = mean_norm * (main_train_rate) ** 0.16
            asr_loss_target = asr_loss_target.detach()
            main_loss_target = main_loss_target.detach()

            optimizer[1].zero_grad()
            loss_sum = torch.add(loss_loss_criterion(asr_norms, asr_loss_target),
                                 loss_loss_criterion(main_norms, main_loss_target))
            loss_sum.backward()

            optimizer[1].step()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer[0].step()
                # scheduler.step()

            loss_value = weighted_loss.item()
            _, top_index = main_output.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs
            correct_sum += (top_index.view(-1) == labels).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info("step: {}\tloss: {:.2f}\ttime(s): {:.2f}".format(step, loss_value, cost_t))

            normalize_coef = 2 / torch.add(model.module.loss_weight.asr_model_weight.data,
                                           model.module.loss_weight.main_model_weight.data)
            model.module.loss_weight.asr_model_weight.data = model.module.loss_weight.asr_model_weight.data * normalize_coef
            model.module.loss_weight.main_model_weight.data = model.module.loss_weight.main_model_weight.data * normalize_coef

            LoggerHelper.info("asr loss weight: {}\tmain loss weight: {}".format(model.module.loss_weight.asr_model_weight.item(), model.module.loss_weight.main_model_weight.item()))

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
                                     shuffle=False)

        predicted_probs = []
        true_labels = []

        batch_count = 1
        for batch in tqdm(eval_dataloader, unit="batch", ncols=100, desc="Evaluating process: "):
            # 获取label和音频数据并配置GPU
            label_inputs = batch['label'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                'label']
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                "asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                'segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else \
            batch["attention_mask"]

            audio_inputs = batch['audio'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                'audio']
            audio_length = batch['audio_length'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else \
            batch['audio_length']

            with torch.no_grad():
                main_output = model(tokens, segment_ids, attention_mask, audio_inputs, audio_length)

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
