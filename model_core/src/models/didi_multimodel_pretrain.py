import time
import yaml
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from model_core.src.utils.log import LoggerHelper
from model_core.src.utils.metric import f1_score

from model_core.src.models.text_bert_embedding_model_all_hidden import TextBertEmbeddingModelAllHiiden
from model_core.src.models.multimodel_attention_pretrain import MultimodelAttention


class DiDiMultimodelPretrain(nn.Module):
    """
    "滴滴"多模态融合模型
    """
    def __init__(self, config_file_path, asr_pretrain_model):
        super(DiDiMultimodelPretrain, self).__init__()

        # 读取配置文件
        with open(config_file_path, 'r') as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

        # 设置文本嵌入维度
        self.asr_embedding_model = TextBertEmbeddingModelAllHiiden(asr_pretrain_model)
        self.asr_embedding_dim = self.asr_embedding_model.output_embedding_dim

        # 设置基于Attention的音频文本对齐网络
        self.mm_attention = MultimodelAttention(config, False, False)

        self.mm_bilstm = nn.LSTM(input_size=20768,
                                 hidden_size=20000,
                                 num_layers=1, batch_first=True, bidirectional=True)

        self.predict_linear = nn.Linear(40000, config['class_num'])

    def forward(self, asr_tokens, asr_segment_ids, asr_attention_mask, text_length, audio_encode, audio_length):
        """

        :param audio_inputs:    音频输入
        :param audio_length:    音频序列实际长度
        :param text_inputs:     文本输入
        :param text_length:     文本序列实际长度
        :return:
        """
        device = audio_encode.device

        # 文本嵌入
        text_encode = self.asr_embedding_model(asr_tokens, asr_segment_ids, asr_attention_mask)

        # 初始化音频mask
        # mask为二维矩阵，每行对应一条样本，为从1到最长样本长度的序列
        audio_mask = torch.arange(audio_encode.size(1))[None, :].repeat(audio_encode.size(0), 1).to(device)
        # 根据实际长度生成元素为0或1的二维矩阵，每行对应一条样本，其中1表示有效元素，0表示Pad造成的无效0元素
        audio_mask = (audio_mask < audio_length[:, None].repeat(1, audio_encode.size(1))).float()
        # 调整mask的维度，增加两个维度对应batch和多头Attention中的不同head
        extended_audio_mask = audio_mask.unsqueeze(1).unsqueeze(2)
        # 将mask中的值转换为0表示有效元素，-10000为Pad后元素
        extended_audio_mask = (1.0 - extended_audio_mask) * -10000.0

        # 初始化文本mask
        # 同上
        text_mask = torch.arange(text_encode.size(1))[None, :].repeat(text_encode.size(0), 1).to(device)
        text_mask = (text_mask < text_length[:, None].repeat(1, text_encode.size(1))).float()

        mm_output = self.mm_attention(text_encode, audio_encode, extended_audio_mask)
        # 将通过对齐操作处理过的音频特征和文本特征拼接
        mm_output = torch.cat([mm_output, text_encode], dim=-1)
        mm_output = mm_output * text_mask[:, :, None]

        # 对音频文本合并特征进行Pack
        mm_pack = nn.utils.rnn.pack_padded_sequence(mm_output, text_length, batch_first=True, enforce_sorted=False)
        # 对合并特征输入双向LSTM进行编码
        mm_output, _ = self.mm_bilstm(mm_pack)
        # 恢复编码结果的Pad形式
        total_length = text_encode.size(1)
        mm_output, _ = nn.utils.rnn.pad_packed_sequence(mm_output, batch_first=True, total_length=total_length)

        # 将Pad部分的值修改为-10000
        # 因为非Pad部分可能出现负值，为了在MaxPool操作时Pad部分不参与运算，需要将其值修改为极小值
        mm_maxpool = mm_output * text_mask[:, :, None] - 10000.0 * (1 - text_mask[:, :, None])
        # MaxPool操作
        mm_maxpool, _ = torch.max(mm_maxpool, dim=1)  # [b,dim]

        # 对音频和文本融合后的特征编码进行FC计算
        logits = self.predict_linear(mm_maxpool)

        return logits

    @ classmethod
    def train_model(cls,
                    master_gpu_id,
                    model,
                    optimizer,
                    scheduler,
                    data_loader,
                    gradient_accumulation_steps,
                    use_cuda):
        model.train()

        loss_function = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        for step, batch in enumerate(data_loader):
            start_t = time.time()

            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]
            tokens_length = batch['tokens_length'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens_length']

            audio_inputs = batch["audio"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["audio"]
            audio_length = batch["audio_length"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["audio_length"]

            prediction = model(tokens, segment_ids, attention_mask, tokens_length, audio_inputs, audio_length)
            loss = loss_function(prediction, labels)

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                scheduler.step()

            loss_value = loss.item()
            _, top_index = prediction.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs
            correct_sum += (top_index.view(-1) == labels).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info("step: {}\tloss: {:.2f}\ttime(s): {:.2f}".format(step, loss, cost_t))

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
            (audio_inputs, text_inputs), label_inputs, _, _ = batch

            label_inputs = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs

            audio_inputs, audio_length = audio_inputs
            audio_inputs = audio_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else audio_inputs
            audio_length = audio_length.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else audio_length

            text_inputs, text_length = text_inputs
            text_inputs = text_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_inputs
            text_length = text_length.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_length

            with torch.no_grad():
                main_output = model(audio_inputs, audio_length, text_inputs, text_length)

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
