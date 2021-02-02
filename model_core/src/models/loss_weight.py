import torch
import torch.nn as nn


class LossWeight(nn.Module):
    """
    用于双任务Loss动态加权求和的权重层
    """
    def __init__(self):
        """
        构造函数
        """
        super(LossWeight, self).__init__()

        # 定义主任务和辅助任务的Loss权重
        # 初始化为1，即总Loss为两个子任务的Loss之和
        self.main_model_weight = nn.Parameter(torch.ones(1).float())
        self.auxiliary_model_weight = nn.Parameter(torch.ones(1).float())

    def forward(self, main_loss, auxiliary_loss):
        """
        前馈函数
        :param main_loss:
        :param auxiliary_loss:
        :return:
        """
        # 求主辅Loss的加权总Loss
        weighted_loss = torch.add(torch.mul(self.auxiliary_model_weight, main_loss),
                                torch.mul(self.main_model_weight, auxiliary_loss))

        return weighted_loss
