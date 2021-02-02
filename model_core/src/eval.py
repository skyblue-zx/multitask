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


def eval_model(task_type,
               master_gpu_id,
               model,
               eval_dataset,
               eval_batch_size=1,
               use_cuda=False,
               num_workers=1):
    """

    :param master_gpu_id:
    :param model:
    :param eval_dataset:
    :param eval_batch_size:
    :param use_cuda:
    :param num_workers:
    :return:
    """
    if task_type == 'single_model_audio':
        AudioSingleModel.eval_model(master_gpu_id=master_gpu_id,
                                    model=model,
                                    eval_dataset=eval_dataset,
                                    eval_batch_size=eval_batch_size,
                                    use_cuda=use_cuda,
                                    num_workers=num_workers)

    elif task_type == 'single_model_audio_lstm':
        AudioSingleModelBasedOnLSTM.eval_model(master_gpu_id=master_gpu_id,
                                               model=model,
                                               eval_dataset=eval_dataset,
                                               eval_batch_size=eval_batch_size,
                                               use_cuda=use_cuda,
                                               num_workers=num_workers)

    elif task_type == 'single_model_text_bert':
        TextSingleModelBasedOnBert.eval_model(master_gpu_id=master_gpu_id,
                                              model=model,
                                              eval_dataset=eval_dataset,
                                              eval_batch_size=eval_batch_size,
                                              use_cuda=use_cuda,
                                              num_workers=num_workers)

    elif task_type == 'single_model_text_gate_merge':
        TextSingleModelAndTextGateMerge.eval_model(master_gpu_id=master_gpu_id,
                                                   model=model,
                                                   eval_dataset=eval_dataset,
                                                   eval_batch_size=eval_batch_size,
                                                   use_cuda=use_cuda,
                                                   num_workers=num_workers)

    elif task_type == 'multimodel_embedding_fuse_text_bert':
        EmbeddingShareMultimodel.eval_model(master_gpu_id=master_gpu_id,
                                            model=model,
                                            eval_dataset=eval_dataset,
                                            eval_batch_size=eval_batch_size,
                                            use_cuda=use_cuda,
                                            num_workers=num_workers)

    elif task_type == 'multimodel_feature_fuse_text_bert_gate_merge':
        OutputGateMergeMultimodel.eval_model(master_gpu_id=master_gpu_id,
                                             model=model,
                                             eval_dataset=eval_dataset,
                                             eval_batch_size=eval_batch_size,
                                             use_cuda=use_cuda,
                                             num_workers=num_workers)

    elif task_type == 'single_model_audio_gate_merge':
        AudioSingleModelAndTextGateMerge.eval_model(master_gpu_id=master_gpu_id,
                                                    model=model,
                                                    eval_dataset=eval_dataset,
                                                    eval_batch_size=eval_batch_size,
                                                    use_cuda=use_cuda,
                                                    num_workers=num_workers)

    elif task_type == 'multimodel_didi':
        DiDiMultimodel.eval_model(master_gpu_id=master_gpu_id,
                                  model=model,
                                  eval_dataset=eval_dataset,
                                  eval_batch_size=eval_batch_size,
                                  use_cuda=use_cuda,
                                  num_workers=num_workers)

    elif task_type == 'multimodel_didi_pretrain':
        pass
        #DiDiMultimodelPretrain.eval_model(master_gpu_id=master_gpu_id,
        #                                  model=model,
        #                                  eval_dataset=eval_dataset,
        #                                  eval_batch_size=eval_batch_size,
        #                                  use_cuda=use_cuda,
        #                                  num_workers=num_workers)

    elif task_type == 'single_model_text_no_pretrain':
        TextSingleModelBasedNoPretrain.eval_model(master_gpu_id=master_gpu_id,
                                                  model=model,
                                                  eval_dataset=eval_dataset,
                                                  eval_batch_size=eval_batch_size,
                                                  use_cuda=use_cuda,
                                                  num_workers=num_workers)

    elif task_type == 'single_model_text_no_pretrain_embedding_share_and_gate_merge':
        TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge.eval_model(master_gpu_id=master_gpu_id,
                                                                            model=model,
                                                                            eval_dataset=eval_dataset,
                                                                            eval_batch_size=eval_batch_size,
                                                                            use_cuda=use_cuda,
                                                                            num_workers=num_workers)

    elif task_type == 'single_model_audio_no_pretrain':
        AudioSingleModelNoPretrain.eval_model(master_gpu_id=master_gpu_id,
                                              model=model,
                                              eval_dataset=eval_dataset,
                                              eval_batch_size=eval_batch_size,
                                              use_cuda=use_cuda,
                                              num_workers=num_workers)

    elif task_type == 'single_model_audio_no_pretrain_gate_merge':
        AudioSingleModelNoPretrainGateMerge.eval_model(master_gpu_id=master_gpu_id,
                                                       model=model,
                                                       eval_dataset=eval_dataset,
                                                       eval_batch_size=eval_batch_size,
                                                       use_cuda=use_cuda,
                                                       num_workers=num_workers)

    elif task_type == 'multimodel_didi_embedding_share':
        DiDiMultimodelEmbeddingShare.eval_model(master_gpu_id=master_gpu_id,
                                                model=model,
                                                eval_dataset=eval_dataset,
                                                eval_batch_size=eval_batch_size,
                                                use_cuda=use_cuda,
                                                num_workers=num_workers)

    elif task_type == 'multimodel_didi_embedding_share_and_output_merge':
        DiDiMultimodelEmbeddingShareAndOutputMerge.eval_model(master_gpu_id=master_gpu_id,
                                                              model=model,
                                                              eval_dataset=eval_dataset,
                                                              eval_batch_size=eval_batch_size,
                                                              use_cuda=use_cuda,
                                                              num_workers=num_workers)

    return
