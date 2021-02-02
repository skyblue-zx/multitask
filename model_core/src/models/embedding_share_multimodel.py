import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss

from model_core.src.models.text_bert_embedding_model import TextBertEmbeddingModel
from model_core.src.models.MobileNetV2_merge_text import mobilenet_v2

from model_core.src.utils.log import LoggerHelper
from model_core.src.utils.metric import f1_score


class EmbeddingShareMultimodel(nn.Module):
    """
    流利度二分类多模态模型
    Embedding共享和多模态模型结果合并
    """
    def __init__(self, asr_pretrain_model, asr_embedding_dim, audio_embedding_dim):
        super(EmbeddingShareMultimodel, self).__init__()

        self.asr_pretrain_model = asr_pretrain_model
        self.asr_embedding_model = TextBertEmbeddingModel(self.asr_pretrain_model, asr_embedding_dim)
        # 设置ASR文本嵌入维度
        self.asr_embedding_dim = self.asr_embedding_model.output_embedding_dim
        # 设置音频嵌入维度
        self.audio_embedding_dim = audio_embedding_dim

        # ASR模型子网络
        self.asr_model = nn.Sequential(
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, 2)
        )

        # 主网络（音频部分）
        self.main_model = mobilenet_v2(2, 768, False)

    def forward(self, asr_tokens, asr_segment_ids, asr_attention_mask, audio):
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
        main_output, audio_embedding = self.main_model(audio, asr_embedding)

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
        model.train()

        loss_criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_sum = 0
        num_batch = data_loader.__len__()
        num_sample = data_loader.dataset.__len__()

        # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
        for step, batch in enumerate(data_loader):
            start_t = time.time()

            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]

            audio = batch["audio"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["audio"]

            main_output, asr_output = model(tokens, segment_ids, attention_mask, audio)

            main_loss = loss_criterion(main_output, labels)
            asr_loss = loss_criterion(asr_output, asr_label)

            overall_loss = main_loss + asr_loss

            if gradient_accumulation_steps > 1:
                overall_loss /= gradient_accumulation_steps

            overall_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                scheduler.step()

            loss_value = overall_loss.item()
            _, top_index = main_output.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = labels.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else labels
            correct_sum += (top_index.view(-1) == labels).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info("step: {}\tloss: {:.2f}\ttime(s): {:.2f}".format(step, overall_loss, cost_t))

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

            audio = batch["audio"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["audio"]

            with torch.no_grad():
                main_output, asr_output = model(tokens, segment_ids, attention_mask, audio)

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


# 仅供单例测试
if __name__ == '__main__':
    # data_path = '/Users/zhaixiao/labeled_data/7763_eng_audio_fluency_20000_outsourcing/test.csv'
    # audio_dir = '/Users/zhaixiao/labeled_data/7763_eng_audio_fluency_20000_outsourcing/media_files_5'
    # max_seq_len = 50
    # asr_pretrain_model = 'bert-base-uncased'
    # audio_pretrain_model = '/Users/zhaixiao/workplace/python/tal_model_train_and_predict_env/model_core/src/models/pytorch_vggish.pth'
    #
    # from model_core.src.data.text_fluency_dataset import TextFluencyDataset
    # dataset = TextFluencyDataset(data_path, audio_dir, max_seq_len, asr_pretrain_model, audio_pretrain_model)
    #
    # from torch.utils.data import DataLoader
    # dataloder = DataLoader(dataset=dataset,
    #                        pin_memory=False,
    #                        batch_size=1,
    #                        num_workers=0,
    #                        shuffle=False)
    # model = EmbeddingShareAndOutputMergeMultimodel(768, 1280)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    #
    # total_steps = dataset.__len__() * 1
    # from transformers import get_linear_schedule_with_warmup
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    #
    # EmbeddingShareAndOutputMergeMultimodel.train_model(master_gpu_id=None,
    #                                                    model=model,
    #                                                    optimizer=optimizer,
    #                                                    scheduler=scheduler,
    #                                                    data_loader=dataloder,
    #                                                    gradient_accumulation_steps=1,
    #                                                    use_cuda=False)

    pass
