import math
import torch
import torch.nn as nn


class MultimodelAttention(nn.Module):
    """
    多模态
    """
    def __init__(self,
                 config,
                 output_attentions=False,
                 keep_multihead_output=False):
        """

        :param config:
        :param output_attentions:
        :param keep_multihead_output:
        """
        super(MultimodelAttention, self).__init__()

        #
        self.output_attentions = output_attentions
        #
        self.keep_multihead_output = keep_multihead_output
        #
        self.multihead_output = None

        # 设置Attention的head个数
        self.num_attention_heads = config['num_attention_heads']
        # 设置每个head的隐层维度
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        # 计算全部head的隐层总维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义Query向量的参数层
        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        # 定义 Key 向量的参数层
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        # 定义Value向量的参数层
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        # 设置Dropout概率
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])

    def transpose_for_scores(self, x):
        """

        :param x:
        :return:
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_hidden_states, attention_mask, head_mask=None):
        """

        :param query_hidden_states:
        :param key_hidden_states:
        :param attention_mask:
        :param head_mask:
        :return:
        """
        # 计算Query向量
        mixed_query_layer = self.query(query_hidden_states)
        # 计算Key向量
        mixed_key_layer = self.key(key_hidden_states)
        # 计算Value向量
        mixed_value_layer = self.value(key_hidden_states)
        # each mixed layer: (batch_size, seqlen, head_num * head_dim)

        # 调整Query、Key和Value的维度格式
        # 将seqlen和对应word的多头Embedding调换位置并将head作为单独维度
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # each layer: (batch_size, head_num, seqlen, head_dim)

        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in TransformerModel forward() function)
        # 加上mask，将padding所在的表示直接-10000
        attention_scores = attention_scores + attention_mask
        # attention_scores: (batch_size, head_num, seqlen, seqlen)

        # 将注意力转化为概率分布，即注意力权重
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        #
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 针对多头Attention不同head的mask
        # 用于认为选择不同head
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 根据Attention Score计算文本的加权Embedding
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: (batch_size, head_num, seqlen, head_dim)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        # 调整文本加权Embedding的维度
        # 将seqlen和head_num两维度交换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 调整文本加权Embedding的维度
        # 将所有head对应的维度合并
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 判断是否输出
        if self.output_attentions:
            return attention_probs, context_layer

        return context_layer
