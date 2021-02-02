import os
import numpy as np
import pandas as pd

from model_core.src.utils.log import LoggerHelper

from model_core.src.prepare import prepare
import model_core.src.train as train
import model_core.src.eval as eval
import model_core.src.predict as predict

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(runtime_config):
    """
    模型环境线下主流程
    :param runtime_config:
    :param call_type: 调用类型：single: 独立调用 interface: 接口调用
    :return:
    """
    ################
    #   运行时参数   #
    ################
    # 获取运行时参数
    # function          运行功能
    # task_type         任务类型
    # mode              运行模式
    # config_name       配置名称
    # gpu_ids           GPU ID配置
    # saved_model       模型参数文件地址
    function = runtime_config.function
    task_type = runtime_config.task_type
    mode = runtime_config.mode
    config_name = runtime_config.config_name
    gpu_ids = runtime_config.gpu_ids
    saved_model = runtime_config.saved_model

    saved_model_list = list()
    
    if saved_model is not None:
        if os.path.isdir(saved_model):
            model_file_list = os.listdir(saved_model)
            for model_file in model_file_list:
                if 'model' in model_file:
                    saved_model_list.append(os.path.join(saved_model, model_file))
        else:
            saved_model_list.append(saved_model)
    else:
        saved_model_list.append(None)

    for saved_model in saved_model_list:
        # 完成模型准备工作
        # 获取配置和模型
        model, dataset, config, master_gpu_id, optimizer, scheduler = prepare(function, task_type, config_name, gpu_ids, saved_model)

        # 根据运行模式进入对应流程
        if mode == 'train':
            ################
            #    训练部分   #
            ################
            LoggerHelper.info("Training".center(60, "="))

            if len(saved_model_list) > 1:
                LoggerHelper.error("The initial model is die".center(60, "="))
                return

            if dataset[0] is not None:
                # 根据Function不同调用不同训练函数
                if function == 'probability':
                    if task_type == 'single_model_text_gate_merge':
                        from model_core.src.train_gate_merge import train_model

                        train_model(task_type,
                                    config["save_path"],
                                    master_gpu_id,
                                    model,
                                    optimizer,
                                    scheduler,
                                    config["epochs"],
                                    dataset[0],
                                    batch_size=config['train_batch_size'],
                                    gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                    use_cuda=config['use_cuda'],
                                    num_workers=config['train_num_workers'],
                                    shuffle=config['train_shuffle'])
                    
                    elif task_type == 'single_model_text_no_pretrain_embedding_share_and_gate_merge':
                        from model_core.src.train_gate_merge_text_no_pretrain import train_model

                        train_model(task_type,
                                    config["save_path"],
                                    master_gpu_id,
                                    model,
                                    optimizer,
                                    scheduler,
                                    config["epochs"],
                                    dataset[0],
                                    batch_size=config['train_batch_size'],
                                    gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                    use_cuda=config['use_cuda'],
                                    num_workers=config['train_num_workers'],
                                    shuffle=config['train_shuffle']) 
                    
                    elif task_type == 'single_model_audio_lstm_text_gate_merge':
                        from model_core.src.train_audio_gate_merge import train_model

                        train_model(task_type,
                                config["save_path"],
                                master_gpu_id,
                                model,
                                optimizer,
                                scheduler,
                                config["epochs"],
                                dataset[0],
                                batch_size=config['train_batch_size'],
                                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                use_cuda=config['use_cuda'],
                                num_workers=config['train_num_workers'],
                                shuffle=config['train_shuffle'])

                    else:
                        train.train_model(task_type,
                                          config["save_path"],
                                          master_gpu_id,
                                          model,
                                          optimizer,
                                          scheduler,
                                          config["epochs"],
                                          dataset[0],
                                          batch_size=config['train_batch_size'],
                                          gradient_accumulation_steps=config["gradient_accumulation_steps"],
                                          use_cuda=config['use_cuda'],
                                          num_workers=config['train_num_workers'],
                                          shuffle=config['train_shuffle'])
                elif function == 'score':
                    pass

            LoggerHelper.info("Training Done".center(60, "="))

        elif mode == 'eval':
            ################
            #    评价部分   #
            ################
            LoggerHelper.info("Evaluating".center(60, "="))
            print(len(dataset[1]))
            if dataset[1] is not None:
                if function == 'probability':
                    eval.eval_model(task_type,
                                    master_gpu_id,
                                    model,
                                    dataset[1],
                                    config["eval_batch_size"],
                                    config["use_cuda"],
                                    config["eval_num_workers"])
                elif function == 'score':
                    pass

            LoggerHelper.info("Evaluating Done".center(60, "="))

        elif mode == 'test':
            pass

        elif mode == 'predict':
            ################
            #    预测部分   #
            ################
            LoggerHelper.info("Predicting".center(60, "="))

            if len(saved_model_list) > 1:
                LoggerHelper.error("The initial model is die".center(60, "="))
                return
            
            if dataset[2] is not None and config['predict_result_save_path']:
                if function == 'probability':
                    predict_results = predict.predict(master_gpu_id,
                                                      model,
                                                      dataset[2],
                                                      config["predict_batch_size"],
                                                      config["use_cuda"],
                                                      config["predict_num_workers"])
                elif function == 'score':
                    predict_results = None
                else:
                    predict_results = None

                predict_result_list = np.array(predict_results)
                result = pd.DataFrame(predict_result_list)
                result.to_csv(config['predict_result_save_path'], index=False, header=False)

            LoggerHelper.info("Predicting Done".center(60, "="))

        # 全部过程结束
        LoggerHelper.info("All process finished.".center(60, "="))

        # return predict_results
