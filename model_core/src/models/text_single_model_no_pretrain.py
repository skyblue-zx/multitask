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


class TextSingleModelBasedNoPretrain(nn.Module):
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
        super(TextSingleModelBasedNoPretrain, self).__init__()

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

        # 设置文本双向LSTM网络
        # 输入维度为文本维度
        self.text_bilstm = nn.LSTM(input_size=self.text_dim, hidden_size=config['text_hidden'],
                                   num_layers=1, batch_first=True, bidirectional=True)

        self.predict_linear = nn.Linear(config['text_hidden'] * 2, config['class_num'])

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
        text_pack = self.text_embeddings(text_inputs)
        # 按照文本序列实际长度对文本序列进行Pack
        text_pack = nn.utils.rnn.pack_padded_sequence(text_pack, text_length, batch_first=True, enforce_sorted=False)
        # 通过双向LSTM对文本编码
        text_encode, _ = self.text_bilstm(text_pack)
        # 恢复文本编码序列的Pad形式
        total_length = text_inputs.size(1)
        text_encode, _ = nn.utils.rnn.pad_packed_sequence(text_encode, batch_first=True, total_length=total_length)

        # 初始化文本mask
        # 同上
        text_mask = torch.arange(text_inputs.size(1))[None, :].repeat(text_inputs.size(0), 1).to(device)
        text_mask = (text_mask < text_length[:, None].repeat(1, text_inputs.size(1))).float()
       
        # 将Pad部分的值修改为-10000
        # 因为非Pad部分可能出现负值，为了在MaxPool操作时Pad部分不参与运算，需要将其值修改为极小值
        text_maxpool = text_encode * text_mask[:, :, None] - 10000.0 * (1 - text_mask[:, :, None])
        # MaxPool操作
        text_maxpool, _ = torch.max(text_maxpool, dim=1)  # [b,dim]

        # 对音频和文本融合后的特征编码进行FC计算
        logits = self.predict_linear(text_maxpool)

        return logits

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

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        for step, batch in enumerate(data_loader):
            start_t = time.time()

            text_inputs, label_inputs, _ = batch

            label_inputs = label_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else label_inputs

            text_inputs, text_length = text_inputs
            text_inputs = text_inputs.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_inputs
            text_length = text_length.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else text_length

            prediction = model(text_inputs, text_length)
            loss = loss_function(prediction, label_inputs)

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
                main_output = model(text_inputs, text_length)

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
