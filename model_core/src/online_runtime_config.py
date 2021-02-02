"""
模型线上配置
"""

import os


class RuntimeConfig:
    # 环境根目录
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # 模型功能
    FUNCTION = 'probability'
    # 配置文件路径
    CONFIG_NAME = 'text_relevance'
    # GPU ID列表
    GPU_IDS = '0'
    # 模型参数文件地址
    SAVED_MODEL = os.path.join(base_dir, 'saved_model', 'text_relevance', 'Epoch_6.model')

    # 预训练模型参数文件目录
    # PRETRAIN_MODEL_DIR = os.path.join(base_dir, 'models', 'pytorch_vggish.pth')
    PRETRAIN_MODEL_DIR = 'bert-base-uncased'
