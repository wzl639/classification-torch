# 数据读取加载类
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode

"""
数据加载
"""


def build_data_set(img_size, data):
    # 构建dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 图片缩放到（224，224）
        transforms.ToTensor(),  # 将图像数据格式转换为Tensor格式。所有数除以255，将数据归一化到[0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准正态分布变换，三通道归一化到[-1,1]。(input-mean)/std
    ])

    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset


if __name__ == '__main__':
    # 测试dataset
    from torch.utils.data import DataLoader

    data = './data/train/'

    # 取训练集合里的图片

    # 数据加载
    train_dataset = build_data_set(224, data)  # 调用dataset.py中的build_data_set()方法
    train_loader = DataLoader(train_dataset, 10, shuffle=True, num_workers=1)

    for i, (images, target) in enumerate(train_loader):
        print(images.shape, target)
        break

    # # 使用迭代器，一次取一个，1个是32个大小
    # data_iter = iter(data_loader)
    # img_tensor, label_tensor = data_iter.__next__()
    #
    # # 打印输出
    # print('batchsize数据集的尺寸集合：', img_tensor.shape)
    # print(img_tensor[9].shape)