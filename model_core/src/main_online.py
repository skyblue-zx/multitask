from model_core.src.utils.log import LoggerHelper

from model_core.src.prepare_online import prepare, dataset_builder
import model_core.src.predict as predict
from model_core.src.online_runtime_config import RuntimeConfig

# 获取运行时配置
runtime_config = RuntimeConfig()

function = runtime_config.FUNCTION
config_name = runtime_config.CONFIG_NAME
gpu_ids = runtime_config.GPU_IDS
saved_model = runtime_config.SAVED_MODEL

pretrain_model_dir = runtime_config.PRETRAIN_MODEL_DIR

model, config, master_gpu_id = prepare(function, config_name, gpu_ids, saved_model, pretrain_model_dir)


def main(data):
    """
    模型环境线上主流程
    :param runtime_config:
    :param call_type: 调用类型：single: 独立调用 interface: 接口调用
    :return:
    """
    print(data)
    ################
    #    预测部分   #
    ################
    LoggerHelper.info("Predicting".center(60, "="))

    dataset = dataset_builder(data, config)

    if function == 'probability':
        predict_results = predict.predict(master_gpu_id,
                                          model,
                                          dataset,
                                          config["predict_batch_size"],
                                          config["use_cuda"],
                                          config["predict_num_workers"])
        LoggerHelper.info("Predict Result: " + str(predict_results))
    elif function == 'score':
        predict_results = None
    else:
        predict_results = None

    LoggerHelper.info("Predicting Done".center(60, "="))

    return predict_results
