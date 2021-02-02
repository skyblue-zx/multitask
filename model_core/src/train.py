import os
import time
import torch
from torch.utils.data import DataLoader

from model_core.src.utils.log import LoggerHelper

from model_core.src.models.audio_single_model import AudioSingleModel
from model_core.src.models.text_single_model_bert import TextSingleModelBasedOnBert
from model_core.src.models.text_single_model_and_text_gate_merge import TextSingleModelAndTextGateMerge
from model_core.src.models.embedding_share_multimodel import EmbeddingShareMultimodel
from model_core.src.models.output_gate_merge_multimodel import OutputGateMergeMultimodel
from model_core.src.models.audio_single_model_and_text_gate_merge import AudioSingleModelAndTextGateMerge
from model_core.src.models.didi_multimodel import DiDiMultimodel
from model_core.src.models.didi_multimodel_embedding_share import DiDiMultimodelEmbeddingShare
from model_core.src.models.didi_multimodel_embedding_share_and_output_merge import DiDiMultimodelEmbeddingShareAndOutputMerge
from model_core.src.models.text_single_model_no_pretrain import TextSingleModelBasedNoPretrain
from model_core.src.models.audio_single_model_no_pretrain import AudioSingleModelNoPretrain
from model_core.src.models.text_single_model_no_pretrain_embedding_share_and_gate_merge import TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge
from model_core.src.models.audio_single_model_no_pretrain_gate_merge import AudioSingleModelNoPretrainGateMerge
#from model_core.src.models.didi_multimodel_pretrain import DiDiMultimodelPretrain
from model_core.src.models.audio_single_model_lstm import AudioSingleModelBasedOnLSTM


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


def train_epoch(master_gpu_id,
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

    total_loss = 0.0
    correct_sum = 0
    num_batch = data_loader.__len__()
    num_sample = data_loader.dataset.__len__()

    # for step, batch in enumerate(tqdm(data_loader, unit="batch", ncols=100, desc="Training process: ")):
    for step, batch in enumerate(data_loader):
        start_t = time.time()

        tokens = batch['tokens'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['tokens']
        segment_ids = batch['segment_ids'].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch['segment_ids']
        attention_mask = batch["attention_mask"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["attention_mask"]
        labels = batch["label"].cuda(master_gpu_id) if use_cuda and master_gpu_id is not None else batch["label"]

        loss, logit = model(tokens, token_type_ids=None, attention_mask=attention_mask, labels=labels)
        loss = loss.mean()

        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
            scheduler.step()

        loss_value = loss.item()
        _, top_index = logit.topk(1)
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

    if task_type in ('multimodel_didi',
                     'multimodel_didi_embedding_share',
                     'multimodel_didi_embedding_share_and_output_merge',
                     'single_model_audio_no_pretrain_gate_merge'):
        from model_core.src.data.didi_dataset import DiDiDataset

        train_loader = DataLoader(dataset=train_dataset,
                                  pin_memory=use_cuda,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle,
                                  collate_fn=DiDiDataset.collate)

    elif task_type in ('single_model_text_no_pretrain',
                       'single_model_text_no_pretrain_embedding_share_and_gate_merge'):
        from model_core.src.data.didi_dataset_text import DiDiDatasetText

        train_loader = DataLoader(dataset=train_dataset,
                                  pin_memory=use_cuda,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle,
                                  collate_fn=DiDiDatasetText.collate)

    elif task_type == 'single_model_audio_no_pretrain':
        from model_core.src.data.didi_dataset_audio import DiDiDatasetAudio

        train_loader = DataLoader(dataset=train_dataset,
                                  pin_memory=use_cuda,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle,
                                  collate_fn=DiDiDatasetAudio.collate)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  pin_memory=use_cuda,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle)

    for epoch in range(1, epochs + 1):
        LoggerHelper.info("Training Epoch: " + str(epoch))

        # avg_loss = train_epoch(master_gpu_id,
        #                        model,
        #                        optimizer,
        #                        scheduler,
        #                        train_loader,
        #                        gradient_accumulation_steps,
        #                        use_cuda)

        if task_type == 'single_model_audio':
            avg_loss = AudioSingleModel.train_model(master_gpu_id,
                                                    model,
                                                    optimizer,
                                                    scheduler,
                                                    train_loader,
                                                    gradient_accumulation_steps,
                                                    use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_audio_lstm':
            avg_loss = AudioSingleModelBasedOnLSTM.train_model(master_gpu_id,
                                                               model,
                                                               optimizer,
                                                               scheduler,
                                                               train_loader,
                                                               gradient_accumulation_steps,
                                                               use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_audio_gate_merge':
            avg_loss = AudioSingleModelAndTextGateMerge.train_model(master_gpu_id,
                                                                    model,
                                                                    optimizer,
                                                                    scheduler,
                                                                    train_loader,
                                                                    gradient_accumulation_steps,
                                                                    use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_text_bert':
            avg_loss = TextSingleModelBasedOnBert.train_model(master_gpu_id,
                                                              model,
                                                              optimizer,
                                                              scheduler,
                                                              train_loader,
                                                              gradient_accumulation_steps,
                                                              use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_text_gate_merge':
            avg_loss = TextSingleModelAndTextGateMerge.train_model(master_gpu_id,
                                                                   model,
                                                                   optimizer,
                                                                   scheduler,
                                                                   train_loader,
                                                                   gradient_accumulation_steps,
                                                                   use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model[0], epoch)

        elif task_type == 'multimodel_embedding_fuse_text_bert':
            avg_loss = EmbeddingShareMultimodel.train_model(master_gpu_id,
                                                            model,
                                                            optimizer,
                                                            scheduler,
                                                            train_loader,
                                                            gradient_accumulation_steps,
                                                            use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_feature_fuse_text_bert_gate_merge':
            avg_loss = OutputGateMergeMultimodel.train_model(master_gpu_id,
                                                             model,
                                                             optimizer,
                                                             scheduler,
                                                             train_loader,
                                                             gradient_accumulation_steps,
                                                             use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_hybrid':
            pass
            # avg_loss = EmbeddingShareAndOutputMergeMultimodel.train_model(master_gpu_id,
            #                                                               model,
            #                                                               optimizer,
            #                                                               scheduler,
            #                                                               train_loader,
            #                                                               gradient_accumulation_steps,
            #                                                               use_cuda)
            #
            # LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            # save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_didi':
            avg_loss = DiDiMultimodel.train_model(master_gpu_id,
                                                  model,
                                                  optimizer,
                                                  scheduler,
                                                  train_loader,
                                                  gradient_accumulation_steps,
                                                  use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_didi_pretrain':
            pass
            #avg_loss = DiDiMultimodelPretrain.train_model(master_gpu_id,
            #                                              model,
            #                                              optimizer,
            #                                              scheduler,
            #                                              train_loader,
            #                                              gradient_accumulation_steps,
            #                                              use_cuda)

        elif task_type == 'single_model_text_no_pretrain':
            avg_loss = TextSingleModelBasedNoPretrain.train_model(master_gpu_id,
                                                                  model,
                                                                  optimizer,
                                                                  scheduler,
                                                                  train_loader,
                                                                  gradient_accumulation_steps,
                                                                  use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_text_no_pretrain_embedding_share_and_gate_merge':
            avg_loss = TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge.train_model(master_gpu_id,
                                                                                            model,
                                                                                            optimizer,
                                                                                            scheduler,
                                                                                            train_loader,
                                                                                            gradient_accumulation_steps,
                                                                                            use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_audio_no_pretrain':
            avg_loss = AudioSingleModelNoPretrain.train_model(master_gpu_id,
                                                              model,
                                                              optimizer,
                                                              scheduler,
                                                              train_loader,
                                                              gradient_accumulation_steps,
                                                              use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'single_model_audio_no_pretrain_gate_merge':
            avg_loss = AudioSingleModelNoPretrainGateMerge.train_model(master_gpu_id,
                                                                       model,
                                                                       optimizer,
                                                                       scheduler,
                                                                       train_loader,
                                                                       gradient_accumulation_steps,
                                                                       use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_didi_embedding_share':
            avg_loss = DiDiMultimodelEmbeddingShare.train_model(master_gpu_id,
                                                                model,
                                                                optimizer,
                                                                scheduler,
                                                                train_loader,
                                                                gradient_accumulation_steps,
                                                                use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

        elif task_type == 'multimodel_didi_embedding_share_and_output_merge':
            avg_loss = DiDiMultimodelEmbeddingShareAndOutputMerge.train_model(master_gpu_id,
                                                                              model,
                                                                              optimizer,
                                                                              scheduler,
                                                                              train_loader,
                                                                              gradient_accumulation_steps,
                                                                              use_cuda)

            LoggerHelper.info("Average Loss: " + format(avg_loss, "0.4f"))
            save_model(model_save_path, model, epoch)

    LoggerHelper.info("Training Done".center(60, "="))

    return
