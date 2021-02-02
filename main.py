import argparse

from model_core.src.main import main


# 总入口
if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Tal model training and predicting enviroment stand-alone running script arguments.")
    parser.add_argument("-f", "--function", dest="function", action="store",
                        help="Running function: probability or score")
    parser.add_argument("-t", "--task_type", dest="task_type", action="store",
                        help="Task type: pre-defined dataset, model type")
    parser.add_argument("-c", "--config", dest="config_name", action="store",
                        help="The name of configuration.")
    parser.add_argument("-m", "--mode", dest="mode", action="store", default="train",
                        help="Running mode: train or eval.")
    parser.add_argument("-g", "--gpu", dest="gpu_ids", action="store",
                        help="Device ids of used gpus, split by ','")
    parser.add_argument("-s", "--saved_model", dest="saved_model", action="store",
                        help="The path of trained checkpoint model.")

    # 获取命令行参数
    parsed_args = parser.parse_args()
    # 启动主流程
    main(parsed_args)
