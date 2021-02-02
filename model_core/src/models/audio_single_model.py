import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from model_core.src.models.MobileNetV2 import mobilenet_v2
from model_core.src.utils.metric import f1_score

from model_core.src.utils.log import LoggerHelper


class AudioSingleModel(nn.Module):
    def __init__(self):
        super(AudioSingleModel, self).__init__()

        self.model = mobilenet_v2(2, False)

    def forward(self, audio):
        output = self.model(audio)

        return output

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

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
        for step, batch in enumerate(data_loader):
            start_t = time.time()

            # 获取label和音频数据并配置GPU
            labels = batch['label'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['label']
            audios = batch['audio'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio']

            # 获取模型输出
            output = model(audios)
            # 计算loss
            loss = loss_criterion(output, labels)
            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps
            # 反向传播
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 清除梯度
                model.zero_grad()
                scheduler.step()

            loss_value = loss.item()
            _, top_index = output.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = labels.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else labels
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
            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
            audio = batch['audio'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio']

            with torch.no_grad():
                output = model(audio)

                # 将模型输出转为列表
                output = torch.softmax(output, dim=1).cpu().tolist()
                # 获取正例结果
                output = np.array(output)[:, 1]
                # 将该Batch的正例预测值列表拼接至全局正例预测值列表中
                predicted_probs.extend(output.tolist())

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
