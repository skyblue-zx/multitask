import torch
import torch.nn as nn


class MultiTaskGate(nn.Module):
    """
    多任务Gate融合模型
    """
    def __init__(self):
        """
        构造函数
        """
        super(MultiTaskGate, self).__init__()

        self.gate_w = nn.Parameter(torch.rand([2, 1]), requires_grad=True)

    def forward(self):
        """
        前馈函数
        生成Gate参数
        :return:
        """
        gate_output = nn.Softmax(dim=0)(self.gate_w)

        return gate_output
