import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from model_core.src.models.text_bert_embedding_model import TextBertEmbeddingModel

from model_core.src.utils.log import LoggerHelper
from model_core.src.utils.metric import f1_score
from model_core.src.models.loss_weight import LossWeight


class TextSingleModelAndTextGateMerge(nn.Module):
    """
    音频文本多模态模型
    Embedding共享和Gate融合
    """
    def __init__(self, asr_pretrain_model, asr_embedding_dim):
        """
        构造函数
        :param asr_pretrain_model:
        :param asr_embedding_dim:
        """
        super(TextSingleModelAndTextGateMerge, self).__init__()

        # 设置文本嵌入模型
        self.asr_embedding_model = TextBertEmbeddingModel(asr_pretrain_model)
        # 获取文本嵌入模型的嵌入维度
        self.asr_embedding_model_dim = self.asr_embedding_model.text_embedding_dim
        # 设置文本嵌入维度
        if asr_embedding_dim > 0:
            self.asr_embedding_dim = asr_embedding_dim
        else:
            self.asr_embedding_dim = self.asr_embedding_model_dim

        # ASR模型子网络
        self.asr_model = nn.Sequential(
            nn.Linear(self.asr_embedding_model_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, 2)
        )

        # 主网络（音频部分）
        self.main_model = nn.Sequential(
            nn.Linear(self.asr_embedding_model_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, 2)
        )

        # Gate网络
        #self.gate = nn.Sequential(
        #    nn.Linear(self.asr_embedding_model_dim, self.asr_embedding_model_dim, bias=True),
        #    nn.ReLU(),
        #    nn.Linear(self.asr_embedding_model_dim, self.asr_embedding_model_dim, bias=True),
        #    nn.ReLU(),
        #    nn.Linear(self.asr_embedding_model_dim, 2, bias=True),
        #    nn.Softmax(dim=1)
        #)
        self.gate_w = torch.nn.Parameter(torch.rand([2, 1]), requires_grad=True)
        self.gate_b = torch.nn.Parameter(torch.randn([1, 2]), requires_grad=True)

    def forward(self, asr_tokens, asr_segment_ids, asr_attention_mask):
        """

        :param sample:  样本对象(包括label asr_label asr_label audio tokens segment_ids attention_mask
        :return:
        """
        # 文本子模型
        asr_embedding = self.asr_embedding_model(asr_tokens, asr_segment_ids, asr_attention_mask)

        # ASR文本子模型输出
        asr_output = self.asr_model(asr_embedding)

        # 主模型
        # 合并ASR文本和音频的特征嵌入作为主模型的输入
        main_output = self.main_model(asr_embedding)

        # 合并全部模型的输出
        #gate_output = self.gate(asr_embedding)
        #gate_output = gate_output.unsqueeze(2)

        merged_output = torch.cat((main_output.unsqueeze(1), asr_output.unsqueeze(1)), 1)
        # gate_output = nn.Softmax(dim=0)(self.gate_w)
        gated_output = merged_output * self.gate_w
        main_output = gated_output.sum(dim=1)

        # 返回主模型输出和ASR文本模型输出
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
        model, loss_weight_model = model[0], model[1]

        model.train()
        loss_weight_model.train()

        loss_criterion = nn.CrossEntropyLoss()
        loss_loss_criterion = nn.L1Loss()

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        if hasattr(model, 'module'):
            current_model = model.module
        else:
            current_model = model

        # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
        for step, batch in enumerate(data_loader):
            start_t = time.time()

            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]

            main_output, asr_output = model(tokens, segment_ids, attention_mask)

            main_loss = loss_criterion(main_output, labels)
            auxiliary_loss = loss_criterion(asr_output, asr_label)

            # 计算加权总损失
            weighted_loss = loss_weight_model(main_loss, auxiliary_loss)
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
                optimizer[0].zero_grad()

            # 损失反向传播且不清除计算图
            weighted_loss.backward(retain_graph=True)
            
            asr_embedding_params = list(current_model.asr_embedding_model.parameters())

            # 计算ASR模型损失到三层FC的梯度
            auxiliary_gradient = torch.autograd.grad(
                auxiliary_loss,
                asr_embedding_params[-2],
                retain_graph=True,
                create_graph=True)
            # 计算上述梯度中对应FC第一层Linear的梯度并加权求二范数
            auxiliary_norms = torch.norm(auxiliary_gradient[0], 2)

            # 计算主模型损失到MobileNetV2的梯度
            main_gradient = torch.autograd.grad(
                main_loss,
                asr_embedding_params[-2],
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
                # scheduler.step()

            loss_value = weighted_loss.item()
            _, top_index = main_output.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = labels.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else labels
            correct_sum += (top_index.view(-1) == labels).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info(
                "step: {}\tloss: {:.2f}\tmain_loss: {:.2f}\tasr_loss: {:.2f}\ttime(s): {:.2f}".format(step, loss_value,
                                                                                                      main_loss,
                                                                                                      auxiliary_loss,
                                                                                                      cost_t))

            normalize_coef = 2/torch.add(loss_weight_model.main_model_weight.data, loss_weight_model.auxiliary_model_weight.data)
            loss_weight_model.auxiliary_model_weight.data = loss_weight_model.auxiliary_model_weight.data * normalize_coef
            loss_weight_model.main_model_weight.data = loss_weight_model.main_model_weight.data * normalize_coef

        LoggerHelper.info(
                "asr loss weight: {}\tmain loss weight: {}".format(
                    loss_weight_model.auxiliary_model_weight.item(),
                    loss_weight_model.main_model_weight.item()))

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
            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]

            tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["tokens"]
            segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["segment_ids"]
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]

            with torch.no_grad():
                main_output, asr_output = model(tokens, segment_ids, attention_mask)

                # 将模型输出转为列表
                main_output = torch.softmax(main_output, dim=1).cpu().tolist()
                # 获取正例结果
                prob = np.array(main_output)[:, 1]
                # 将该Batch的正例预测值列表拼接至全局正例预测值列表中
                predicted_probs.extend(prob.tolist())

                # 将真实label列表拼接至全局真实label列表
                true_labels.extend(labels.tolist())

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
