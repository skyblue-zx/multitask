# 加载通用依赖库
import os
import yaml
import torch
import torchtext.vocab as vocab
#from model_core.src.utils.task_router import TaskRouter
# 加载日志模块
from model_core.src.utils.log import logger_config
from model_core.src.utils.log import LoggerHelper

# 加载个性化依赖库
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

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
from model_core.src.models.loss_weight import LossWeight
from model_core.src.models.audio_single_model_lstm_text_gate_merge import AudioSingleModelBasedOnLSTMTextGateMerge

# 加载个性化数据集
from model_core.src.data.text_relevance_dataset import TextRelevanceDataset
from model_core.src.data.fluency_dataset import FluencyDataset
from model_core.src.data.didi_dataset import DiDiDataset
from model_core.src.data.didi_dataset_text import DiDiDatasetText
from model_core.src.data.didi_dataset_audio import DiDiDatasetAudio


# 获取model_core的根目录路径
base_dir = os.path.dirname(os.path.dirname(__file__))

torch.backends.cudnn.enabled = False


def prepare(function, task_type, config_name, gpu_ids, saved_model):
    """
    模型准备函数
    完成模型训练、验证和预测的各项准备工作
    :param function:        执行功能名称
    :param config_name:     配置名称
    :param gpu_ids:         GPU ID列表字符串（逗号分隔）
    :param saved_model:     模型保存目录名称
    :return:
    """
    ################
    #    配置部分   #
    ################

    # 根据Function的不同加载对应配置文件
    config_file_path = os.path.join(base_dir, 'config') + '/' + config_name + '.' + function + '.yaml'
    with open(config_file_path, 'r') as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    # 设置模型保存路径
    config['save_path'] = os.path.join(base_dir, 'saved_model', config['instance_name'])

    ################
    #    日志部分   #
    ################
    # 获取日志文件路径
    log_file_path = os.path.join(base_dir, 'log', config['instance_name'] + '_' + str(function) + '.log')
    # 配置日志系统
    logger_config(log_file_name=log_file_path, log_level=config['log_level'], need_loghead=False, timed_rotating=True)
    LoggerHelper.info("Loading HyperParameters".center(60, "="))
    LoggerHelper.info(config)
    LoggerHelper.info("Load HyperParameters Done".center(60, "="))

    ################
    #    GPU配置    #
    ################
    # 根据配置文件设置是否使用GPU标识
    use_cuda = config["use_cuda"]
    # 初始化主GPU ID为空
    master_gpu_id = None
    # 初始化GPU ID列表为空
    gpu_id_list = None
    if gpu_ids:
        if len(gpu_ids) == 1:
            master_gpu_id = int(gpu_ids)
        else:
            gpu_id_list = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
            master_gpu_id = gpu_id_list[0]

    ################
    #    模型部分   #
    ################
    # 初始化模型
    if function == 'probability':
        # Todo
        # 根据任务路由和统一配置文件完成基于不同任务的统一动态调用
        # task_router = TaskRouter()
        # task_dict =

        if task_type == 'single_model_audio':
            model = AudioSingleModel()

        elif task_type == 'single_model_audio_lstm':
            model = AudioSingleModelBasedOnLSTM(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])
        
        elif task_type == 'single_model_audio_lstm_text_gate_merge':
            model = AudioSingleModelBasedOnLSTMTextGateMerge(
                    config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'],
                    asr_pretrain_model=config['text_pretrain_model'],
                    asr_embedding_dim=768,
                    audio_embedding_dim=12288)

        elif task_type == 'single_model_audio_gate_merge':
            model = AudioSingleModelAndTextGateMerge(config['text_pretrain_model'],
                                                     asr_embedding_dim=768,
                                                     audio_pretrain_model=config['audio_pretrain_model'],
                                                     audio_embedding_dim=1280)

        elif task_type == 'single_model_text_bert':
            model = TextSingleModelBasedOnBert(config['text_pretrain_model'])
            # model = BertForSequenceClassification.from_pretrained(config["text_pretrain_model_dir"],
            #                                                   num_labels=config["num_labels"],
            #                                                   output_attentions=False,  # 模型是否返回 attentions weights.
            #                                                   # output_hidden_states = False, # 模型是否返回所有隐层状态.
            #     )

        elif task_type == 'single_model_text_gate_merge':
            model = TextSingleModelAndTextGateMerge(config['text_pretrain_model'],
                                                    asr_embedding_dim=768)

        elif task_type == 'multimodel_embedding_fuse_text_bert':
            model = EmbeddingShareMultimodel(config['text_pretrain_model'],
                                             asr_embedding_dim=768,
                                             audio_embedding_dim=1280)

        elif task_type == 'multimodel_feature_fuse_text_bert_gate_merge':
            model = OutputGateMergeMultimodel(config['text_pretrain_model'],
                                              asr_embedding_dim=768,
                                              audio_embedding_dim=1280)

        # elif task_type == 'multimodel_hybrid':
        #     model = EmbeddingShareAndOutputMergeMultimodel(config['text_pretrain_model'],
        #                                                    asr_embedding_dim=768,
        #                                                    audio_pretrain_model=config['audio_pretrain_model'],
        #                                                    audio_embedding_dim=1280)

        elif task_type == 'multimodel_didi':
            model = DiDiMultimodel(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'multimodel_didi_pretrain':
            pass
            #model = DiDiMultimodelPretrain(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'],
            #asr_pretrain_model=config['text_pretrain_model'])

        elif task_type == 'single_model_text_no_pretrain':
            model = TextSingleModelBasedNoPretrain(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'single_model_text_no_pretrain_embedding_share_and_gate_merge':
            model = TextSingleModelBasedNoPretrainEmbeddingShareAndGateMerge(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'single_model_audio_no_pretrain':
            model = AudioSingleModelNoPretrain(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'single_model_audio_no_pretrain_gate_merge':
            model = AudioSingleModelNoPretrainGateMerge(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'multimodel_didi_embedding_share':
            model = DiDiMultimodelEmbeddingShare(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

        elif task_type == 'multimodel_didi_embedding_share_and_output_merge':
            model = DiDiMultimodelEmbeddingShareAndOutputMerge(config_file_path=os.path.join(base_dir, 'config') + '/' + config['didi_multimodel_config_file'])

    elif function == 'score':
        model = None

    # 判断是否使用GPU
    if use_cuda and master_gpu_id is not None:
        # 判断是否加载已有模型
        if saved_model:
            LoggerHelper.info("Loading Saved Model".center(60, "="))
            LoggerHelper.info("Load saved model from: " + saved_model)
            # model.load_state_dict(torch.load(saved_model))
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(saved_model).items()})
            LoggerHelper.info("Loading Saved Model Done".center(60, "="))

        LoggerHelper.info("GPU training or evaluating.")
        model = model.cuda(int(master_gpu_id))
        # 判断是否使用多GPU
        if gpu_id_list is not None:
            LoggerHelper.info("Multiple GPU training or evaluating.")
            model = torch.nn.DataParallel(model, device_ids=gpu_id_list)
        else:
            LoggerHelper.info("Single GPU training or evaluating.")
    else:
        # 判断是否加载已有模型
        if saved_model:
            LoggerHelper.info("Loading Saved Model".center(60, "="))
            LoggerHelper.info("Load saved model from: " + saved_model)
            model.load_state_dict(torch.load(saved_model, map_location='cpu'))
            LoggerHelper.info("Loading Saved Model Done".center(60, "="))

    ################
    #    数据部分   #
    ################
    LoggerHelper.info("Loading Dataset".center(60, "="))
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    # 根据Function不同加载对应数据集
    if function == 'probability':
        if config['train_dataset_path'] and os.path.exists(config['train_dataset_path']):
            if task_type in ('multimodel_didi',
                             'multimodel_didi_embedding_share',
                             'multimodel_didi_embedding_share_and_output_merge',
                             'single_model_audio_no_pretrain_gate_merge'):
                glove = vocab.GloVe(name='6B', dim=300,
                                    cache=config['didi_multimodel_vocabulary_dict'])
                vocabulary_dict = glove.stoi

                train_dataset = DiDiDataset(data=config['train_dataset_path'],
                                            audio_dir=config['train_audio_dir'],
                                            vocabulary_dict=vocabulary_dict,
                                            audio_length=config['didi_multimodel_audio_length'])
            elif task_type in ('single_model_text_no_pretrain',
                               'single_model_text_no_pretrain_embedding_share_and_gate_merge'):
                glove = vocab.GloVe(name='6B', dim=300,
                                    cache=config['didi_multimodel_vocabulary_dict'])
                vocabulary_dict = glove.stoi

                train_dataset = DiDiDatasetText(data=config['train_dataset_path'],
                                                vocabulary_dict=vocabulary_dict)

            elif task_type == 'single_model_audio_no_pretrain':
                train_dataset = DiDiDatasetAudio(data=config['train_dataset_path'],
                                                 audio_dir=config['train_audio_dir'],
                                                 audio_length=config['didi_multimodel_audio_length'])

            else:
                train_dataset = FluencyDataset(data=config['train_dataset_path'],
                                               task_type=task_type,
                                               audio_dir=config['train_audio_dir'],
                                               max_seq_len=config['max_seq_len'],
                                               asr_pretrain_model=config['text_pretrain_model'],
                                               audio_pretrain_model=config['audio_pretrain_model'],
                                               predict=False,
                                               cache=config['cache'],
                                               temp_dir=config['temp_dir'])

        if config['eval_dataset_path'] and os.path.exists(config['eval_dataset_path']):
            if task_type in ('multimodel_didi',
                             'multimodel_didi_embedding_share',
                             'multimodel_didi_embedding_share_and_output_merge',
                             'single_model_audio_no_pretrain_gate_merge'):
                glove = vocab.GloVe(name='6B', dim=300,
                                    cache=config['didi_multimodel_vocabulary_dict'])
                vocabulary_dict = glove.stoi

                eval_dataset = DiDiDataset(data=config['eval_dataset_path'],
                                           audio_dir=config['eval_audio_dir'],
                                           vocabulary_dict=vocabulary_dict,
                                           audio_length=config['didi_multimodel_audio_length'])

            elif task_type in ('single_model_text_no_pretrain',
                               'single_model_text_no_pretrain_embedding_share_and_gate_merge'):
                glove = vocab.GloVe(name='6B', dim=300,
                                    cache=config['didi_multimodel_vocabulary_dict'])
                vocabulary_dict = glove.stoi

                eval_dataset = DiDiDatasetText(data=config['eval_dataset_path'],
                                               vocabulary_dict=vocabulary_dict)

            elif task_type == 'single_model_audio_no_pretrain':
                eval_dataset = DiDiDatasetAudio(data=config['eval_dataset_path'],
                                                 audio_dir=config['train_audio_dir'],
                                                 audio_length=config['didi_multimodel_audio_length'])

            else:
                eval_dataset = FluencyDataset(data=config['eval_dataset_path'],
                                              task_type=task_type,
                                              audio_dir=config['eval_audio_dir'],
                                              max_seq_len=config['max_seq_len'],
                                              asr_pretrain_model=config['text_pretrain_model'],
                                              audio_pretrain_model=config['audio_pretrain_model'],
                                              predict=False,
                                              cache=config['cache'],
                                              temp_dir=config['temp_dir'])

        if config['predict_dataset_path'] and os.path.exists(config['predict_dataset_path']):
            pass
            # predict_dataset = AudioFluencyDataset(data=config['predict_dataset_path'],
            #                                       audio_dir=config['predict_audio_path'],
            #                                       max_seq_len=config['max_seq_len'],
            #                                       audio_pretrain_model_dir=config['audio_pretrain_model_dir'],
            #                                       text_pretrain_model_dir=config['text_pretrain_model_dir'])
    elif function == 'score':
        pass

    LoggerHelper.info("Loading Dataset Done".center(60, "="))

    ################
    #   优化器部分   #
    ################
    optimizer = None

    if task_type == 'single_model_audio_gate_merge':
        pass
        # loss_params_id = list()
        # loss_params = list()
        #
        # from model_core.src.models.loss_weight import LossWeight
        # for m in model.modules():
        #     if isinstance(m, LossWeight):
        #         loss_params_id += list(map(id, m.parameters()))
        #         loss_params += m.parameters()
        #
        # base_params = list(filter(lambda p: id(p) not in loss_params_id, model.parameters()))
        #
        # base_optimizer = AdamW(base_params, lr=config['lr'])
        # loss_optimizer = AdamW(loss_params, lr=config['lr'])
        # optimizer = [base_optimizer, loss_optimizer]

    else:
        optimizer = AdamW(model.parameters(),
                          lr=config['lr'], # args.learning_rate - default is 5e-5
                          # eps = 1e-8 # args.adam_epsilon  - default is 1e-8
        )

    ################
    #   调度器部分   #
    ################
    scheduler = None

    if task_type == 'single_model_audio_gate_merge':
        pass

    elif task_type in ('single_model_text_no_pretrain_embedding_share_and_gate_merge',
                       'single_model_audio_no_pretrain_gate_merge',
                       'single_model_text_gate_merge',
                       'single_model_audio_lstm_text_gate_merge'):
        pass

    else:
        total_steps = None
        if train_dataset is not None:
            total_steps = train_dataset.__len__() * config["epochs"]
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return model, [train_dataset, eval_dataset, predict_dataset], config, master_gpu_id, optimizer, scheduler
