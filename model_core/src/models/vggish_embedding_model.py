import torch
import torch.nn as nn
from model_core.src.models.vggish import VGGish


class VGGishEmbeddingModel(nn.Module):
    """
    VGGish嵌入模型
    """
    def __init__(self, audio_pretrain_model):
        super(VGGishEmbeddingModel, self).__init__()

        # 设置音频预训练模型
        self.audio_pretrain_model = audio_pretrain_model

        self.vggish_model = VGGish()
        pytorch_model_state = self.vggish_model.state_dict()
        checkpoint = torch.load(self.audio_pretrain_model)
        state_dict = {k: v for k, v in checkpoint.items() if k in pytorch_model_state.keys()}
        pytorch_model_state.update(state_dict)
        self.vggish_model.load_state_dict(pytorch_model_state)

    def forward(self, raw_audio):
        # 获得模型输出
        audio = self.vggish_model(raw_audio)
        # 截取60秒音频
        audio = audio[:60 if audio.shape[0] > 60 else audio.shape[0], :]
        # 打平音频多维向量
        audio = audio.view(audio.size(0) * audio.size(1))
        # 计算pad个数
        pad_size = 737280 - audio.size(0)
        # 对不足60秒的音频进行pad
        audio = torch.cat([audio, torch.zeros(pad_size)])
        # 恢复音频多维向量
        audio = audio.view(-1, 512 * 24)
        # 增加维度生成符合Batch的数据格式
        audio = audio.unsqueeze(dim=0)

        return audio
