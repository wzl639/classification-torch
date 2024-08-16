import torch
import torchvision
from torch import nn

"""
模型，这边直接使用了预训练的resnet18,可以换成自己搭建的模型
"""


def buid_model(num_class):
    """
    构建一个分类模型
    :param num_class: 分类模型的类别数
    :return: 分类模型
    """
    # 构建一个分类模型
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)  # 将全连接层输出设置为自己的类别数
    return model


if __name__ == '__main__':
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 224, 224))
    target = torch.tensor([[1, 0]])
    target = target.to(device)
    print(target)
    x = x.to(device)
    model = buid_model(2)
    model.to(device)
    # print(model)
    y = model(x)
    print(y.shape, y)
    _, pred = torch.max(y.data, 1)
    _, target = torch.max(target.data, 1)