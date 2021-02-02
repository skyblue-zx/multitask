import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_core.src.utils.log import LoggerHelper

from model_core.src.models.audio_single_model_lstm_text_gate_merge import AudioSingleModelBasedOnLSTMTextGateMerge


def save_model(save_dir, model, epoch):
    """
    保存模型
    :param save_dir:   保存路径
    :param model:       模型
    :param epoch:       训练Epoch
    :return:
    """
    LoggerHelper.info("Save Model".center(60, "="))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = 'Epoch_' + str(epoch) + '.model'
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)

    LoggerHelper.info("Save Model Done".center(60, "="))
    return


def train_model(task_type,
                model_save_path,
                master_gpu_id,
                model,
                optimizer,
                scheduler,
                epochs,
                train_dataset,
                batch_size,
                gradient_accumulation_steps=1,
                use_cuda=False,
                num_workers=1,
                shuffle=True):
    """
    模型训练
    :param model_save_path:
    :param master_gpu_id:
    :param model:
    :param optimizer:
    :param scheduler:
    :param epochs:
    :param train_dataset:
    :param batch_size:
    :param gradient_accumulation_steps:
    :param use_cuda:
    :param num_workers:
    :param shuffle:
    :return:
    """
    LoggerHelper.info("Start Training".center(60, "="))

    data_loader = DataLoader(dataset=train_dataset,
                              pin_memory=use_cuda,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=shuffle)

    model.train()

    loss_criterion = nn.CrossEntropyLoss()
    loss_loss_criterion = nn.L1Loss()

    num_batch = data_loader.__len__()
    num_sample = data_loader.dataset.__len__()

    device = torch.device('cuda:' + str(master_gpu_id) if use_cuda and master_gpu_id is not None else 'cpu')
    main_loss_weight = torch.tensor([1], dtype=torch.float64, requires_grad=True, device=device)
    auxiliary_loss_weight = torch.tensor([1], dtype=torch.float64, requires_grad=True, device=device)

    loss_params = [main_loss_weight, auxiliary_loss_weight]
    loss_optimizer = torch.optim.Adam(loss_params, lr=optimizer.state_dict()['param_groups'][0]['lr'])

    for epoch in range(1, epochs + 1):
        LoggerHelper.info("Training Epoch: " + str(epoch))

        total_loss = 0.0
        correct_sum = 0

        if hasattr(model, 'module'):
            current_model = model.module
        else:
            current_model = model

        # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
        for step, batch in enumerate(data_loader):
            start_t = time.time()

            labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]
            asr_label = batch["asr_label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                "asr_label"]

            tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
            segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch[
                'segment_ids']
            attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else \
            batch["attention_mask"]
            
            audio_inputs = batch['audio'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio']
            audio_length = batch['audio_length'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['audio_length']

            main_output, asr_output = model(tokens, segment_ids, attention_mask, audio_inputs, audio_length)

            main_loss = loss_criterion(main_output, labels)
            auxiliary_loss = loss_criterion(asr_output, asr_label)

            # 计算加权总损失
            weighted_main_loss = loss_params[0] * main_loss
            weighted_auxiliary_loss = loss_params[1] * auxiliary_loss
            # weighted_loss = torch.div(torch.add(weighted_main_loss, weighted_auxiliary_loss), 2)
            weighted_loss = torch.div(torch.add(weighted_main_loss, weighted_auxiliary_loss), 2)
            # 求加权损失累积均值
            if gradient_accumulation_steps > 1:
                weighted_loss /= gradient_accumulation_steps

            # 记录第一个Batch的损失
            # 用于之后计算评估训练速度的参数
            if step == 0:
                initial_auxiliary_loss = auxiliary_loss.detach()
                initial_main_loss = main_loss.detach()

            # # 判断是否到达累计总步数
            # # 若到达则清空累计梯度
            # if step % gradient_accumulation_steps == 0:
            #     optimizer.zero_grad()

            # 损失反向传播且不清除计算图
            optimizer.zero_grad()
            weighted_loss.backward(retain_graph=True)
            optimizer.step()

            asr_embedding_params = list(current_model.asr_embedding_model.parameters())

            # 计算ASR模型损失到三层FC的梯度
            auxiliary_gradient = torch.autograd.grad(
                weighted_auxiliary_loss,
                asr_embedding_params[-2],
                retain_graph=True,
                create_graph=True)
            # 计算上述梯度中对应FC第一层Linear的梯度并加权求二范数
            auxiliary_norms = torch.norm(auxiliary_gradient[0], 2)

            # 计算主模型损失到MobileNetV2的梯度
            main_gradient = torch.autograd.grad(
                weighted_main_loss,
                asr_embedding_params[-2],
                retain_graph=True,
                create_graph=True)
            # 计算上述梯度中对应MobileNetV2第一层的梯度并加权求二范数
            main_norms = torch.norm(main_gradient[0], 2)

            # 求上述两个梯度二范数的均值
            mean_norm = torch.div(torch.add(auxiliary_norms, main_norms), 2)

            # 计算ASR模型当前Batch loss与首个Batch loss的比例
            auxiliary_loss_ratio = torch.div(auxiliary_loss, initial_auxiliary_loss)
            # 计算主模型当前Batch loss与首个Batch loss的比例
            main_loss_ratio = torch.div(main_loss, initial_main_loss)
            mean_loss_ratio = torch.div(torch.add(auxiliary_loss_ratio, main_loss_ratio), 2)

            # 计算ASR模型当前的训练速度参数
            auxiliary_train_rate = torch.div(auxiliary_loss_ratio, mean_loss_ratio)
            # 计算主模型当前的训练速度参数
            main_train_rate = torch.div(main_loss_ratio, mean_loss_ratio)

            # Todo
            # 超参读入
            auxiliary_loss_target = mean_norm * (auxiliary_train_rate) ** 0.12
            main_loss_target = mean_norm * (main_train_rate) ** 0.12
            auxiliary_loss_target = auxiliary_loss_target.detach()
            main_loss_target = main_loss_target.detach()

            loss_optimizer.zero_grad()
            loss_sum = torch.add(loss_loss_criterion(auxiliary_norms, auxiliary_loss_target),
                                 loss_loss_criterion(main_norms, main_loss_target))
            loss_sum.backward()

            loss_optimizer.step()

            # if (step + 1) % gradient_accumulation_steps == 0:
            #     optimizer.step()
            #     # scheduler.step()

            loss_value = weighted_loss.item()
            _, top_index = main_output.topk(1)
            top_index = top_index.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else top_index
            labels = labels.cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else labels
            correct_sum += (top_index.view(-1) == labels).sum().item()
            total_loss += loss_value

            cost_t = time.time() - start_t
            LoggerHelper.info(
                "step: {}\tloss: {:.2f}\tmain_loss: {:.2f}\tasr_loss: {:.2f}\ttime(s): {:.2f}".format(step, 
                                                                                                      loss_value,
                                                                                                      main_loss.item(),
                                                                                                      auxiliary_loss.item(),
                                                                                                      cost_t))

            coef = 2 / torch.add(main_loss_weight, auxiliary_loss_weight)
            loss_params = [coef * main_loss_weight, coef * auxiliary_loss_weight]

            LoggerHelper.info("main loss weight: {}\tauxiliary loss weight: {}".format(main_loss_weight.item(), auxiliary_loss_weight.item()))

        LoggerHelper.info("Total Training Samples: " + str(num_sample))
        LoggerHelper.info("Correct Prediction: " + str(correct_sum))
        LoggerHelper.info("Error Rate: " + format(1 - (correct_sum / num_sample), "0.4f"))

        avg_loss = total_loss / num_batch

        LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
        save_model(model_save_path, model, epoch)

    LoggerHelper.info("Training Done".center(60, "="))

    return
