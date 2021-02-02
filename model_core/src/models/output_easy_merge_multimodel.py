import torch
import torch.nn as nn

from model_core.src.models.text_bert_embedding_model import TextBertEmbeddingModel
from model_core.src.models.MobileNetV2 import mobilenet_v2


class OutputMergeMultiodel(nn.Module):
    def __init__(self, asr_pretrain_model, asr_embedding_dim, audio_embedding_dim):
        super(OutputMergeMultiodel, self).__init__()

        self.asr_pretrain_model = asr_pretrain_model
        self.asr_embedding_model = TextBertEmbeddingModel(self.asr_pretrain_model, asr_embedding_dim)
        # 设置ASR文本嵌入维度
        self.asr_embedding_dim = self.asr_embedding_model.output_embedding_dim
        # 设置音频嵌入维度
        self.audio_embedding_dim = audio_embedding_dim

        # ASR模型子网络
        self.asr_model = nn.Sequential(
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, self.asr_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.asr_embedding_dim, 2)
        )

        # 音频模型子网络
        self.audio_model = mobilenet_v2(2, False)

    def forward(self, asr_tokens, asr_segment_ids, asr_attention_mask, audio):
        # 文本子模型
        asr_embedding = self.asr_embedding_model(asr_tokens, asr_segment_ids, asr_attention_mask)

        # ASR文本子模型输出
        asr_output = self.asr_model(asr_embedding)

        # 音频模型
        audio_output = self.audio_model(audio, asr_embedding)

        # 合并全部模型的输出
        all_embedding = torch.cat((asr_embedding, audio_embedding), 1)

        return output[1]
