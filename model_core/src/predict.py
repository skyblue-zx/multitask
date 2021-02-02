import torch
import numpy as np
from torch.utils.data import DataLoader

from model_core.src.utils.log import LoggerHelper


def predict(master_gpu_id,
            model,
            predict_dataset,
            predict_batch_size=1,
            use_cuda=False,
            num_workers=1):
    """

    :param master_gpu_id:
    :param model:
    :param predict_dataset:
    :param predict_batch_size:
    :param use_cuda:
    :param num_workers:
    :return:
    """
    LoggerHelper.info("Start Predicing".center(60, "="))

    # 设置模型为评估状态
    model.eval()
    # 加载预测数据加载器
    predict_loader = DataLoader(dataset=predict_dataset,
                                pin_memory=use_cuda,
                                batch_size=predict_batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    # 初始化模型预测结果列表
    predicted_result_list = list()

    # 遍历评估数据集中的每个batch
    current_batch_index = 0
    for batch in predict_loader:
        LoggerHelper.info("Batch: " + str(current_batch_index))
        # current_batch_index += 1
        #
        # tokens = batch["tokens"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["tokens"]
        # segment_ids = batch["segment_ids"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
        #     "segment_ids"]
        # attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else \
        # batch["attention_mask"]
        # labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
        #
        # # 获取模型输出
        # with torch.no_grad():
        #     _, logit = model(tokens,
        #                      token_type_ids=None,
        #                      attention_mask=attention_mask,
        #                      labels=labels)
        #
        # # 将模型输出转为列表
        # logit = torch.softmax(logit, dim=1).cpu().tolist()
        # # 获取正例结果
        # logit = np.array(logit)[:, 1]
        # predicted_result_list.extend(logit.tolist())

    LoggerHelper.info("Predicting Ends".center(60, "="))

    return predicted_result_list
