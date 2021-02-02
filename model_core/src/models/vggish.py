import torch
import torch.nn as nn


class VGGish(nn.Module):
    """
    VGGish模型
    """
    def __init__(self):
        super(VGGish, self).__init__()

        # 特征处理部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        # 全连接部分
        # self.fc = nn.Sequential(
        #     nn.Linear(512 * 24, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 128),
        #     nn.ReLU(inplace=True),
        # )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # print(x.shape)
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        # x = self.features(x)
        # print(x.shape)
        # x = x.permute(0, 2, 3, 1).contiguous()
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.fc(x)
        # print(x.shape)

        return x


def main():
    """
    仅供单元测试
    :return:
    """
    # Initialize the PyTorch model.
    # device = 'cuda:0'
    pytorch_model = VGGish()
    pytorch_model.load_state_dict(torch.load('pytorch_vggish.pth'))
    # pytorch_model = pytorch_model.to(device)
    # print(pytorch_model.state_dict())


# 仅供单元测试
if __name__ == '__main__':
    main()
