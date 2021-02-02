import torch.nn as nn
from transformers import BertModel


class TextBertEmbeddingModel(nn.Module):
    """
    文本Bert嵌入模型
    """
    def __init__(self, pretrain_model):
        """
        构造函数
        :param pretrain_model:
        """
        super(TextBertEmbeddingModel, self).__init__()

        # 设置预训练模型
        self.pretrain_model = pretrain_model
        # 获取嵌入器
        self.embedder = BertModel.from_pretrained(pretrain_model)
        # 获取嵌入维度
        # 该模型作为预训练模型发挥作用，所以模型参数通过文件加载，模型结构无法修改
        self.text_embedding_dim = self.embedder.config.hidden_size

    def forward(self, tokens, segments, input_masks):
        """
        前馈函数
        :param tokens:
        :param segments:
        :param input_masks:
        :return:
        """
        # output为有两个元素的tuple
        # output[0]为last_hidden_state
        # output[1]为pooler_output，即最后一层CLS对应的向量
        # output[1]的shape为B * S * D
        # 其中：
        # B-batch
        # S-Seq
        # D-Embedding长度，默认768
        output = self.embedder(tokens, token_type_ids=segments, attention_mask=input_masks)

        return output[1]
