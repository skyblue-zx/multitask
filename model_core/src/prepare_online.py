# 加载通用依赖库
import os
import yaml
import torch
# 加载日志模块
from model_core.src.utils.log import logger_config
from model_core.src.utils.log import LoggerHelper

# 加载个性化依赖库
from transformers import AdamW
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

# 加载个性化数据集
from model_core.src.data.text_relevance_dataset import TextRelevanceDataset


# 获取model_core的根目录路径
base_dir = os.path.dirname(os.path.dirname(__file__))


def prepare(function, config_name, gpu_ids, saved_model, pretrain_model=''):
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
    # 设置预训练模型保存路径
    if pretrain_model != '':
        config["pretrain_model_dir"] = pretrain_model

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
        model = BertForSequenceClassification.from_pretrained(config["pretrain_model_dir"],
                                                              num_labels=config["num_labels"],
                                                              output_attentions=False,  # 模型是否返回 attentions weights.
                                                              # output_hidden_states = False, # 模型是否返回所有隐层状态.
            )
    elif function == 'score':
        model = None

    # 判断是否使用GPU
    if use_cuda:
        # 判断是否设置主GPU ID
        if master_gpu_id is not None:
            # 判断是否加载已有模型
            if saved_model:
                LoggerHelper.info("Loading Saved Model".center(60, "="))
                LoggerHelper.info("Load saved model from: " + saved_model)
                model.load_state_dict(torch.load(saved_model))
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

    return model, config, master_gpu_id


def dataset_builder(data, config):
    dataset = TextRelevanceDataset(data=data,
                                           max_seq_len=config['max_seq_len'],
                                           model_dir=config['pretrain_model_dir'])

    return dataset
