from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse
from dataset import build_data_set
from model import buid_model
import os
from tqdm import tqdm

"""
模型训练，验证脚本
"""

# 训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def val(model, device, test_loader, criterion):
    """
    模型验证，没训练完一个epoch计算模型在测试集上的准确率和loss
    :param model: 训练的模型
    :param device: 设备
    :param test_loader: 测试集数据加载类
    :param criterion: 损失函数
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader)
    # print(total_num)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 统计loss
            loss = criterion(output, target)
            print_loss = loss.data.item()
            test_loss += print_loss
            # 统计准确率
            _, pred = torch.max(output.data, 1)
            # _, label = torch.max(target.data, 1)
            correct += 1 if int(pred) == int(target) else 0

        acc = correct / total_num
        avgloss = test_loss / total_num
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, total_num, 100 * acc))


def main(args):
    # 数据加载
    train_dataset = build_data_set(args.image_size, args.train_data)  # 调用dataset.py中的build_data_set()方法
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 测试集数加载
    test_dataset = build_data_set(args.image_size, args.test_data)  # 调用dataset.py中的build_data_set()方法
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # 模型
    model = buid_model(args.classes_num)
    model.to(device)

    # 损失函数选择
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # 正式训练

    for epoch in range(args.epochs):  # args.epochs，epochs=10
        print(f'Epoch {epoch}/{args.epochs}')
        # switch to train mode
        model.train()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.to(device), target.to(device)
            # 计算输出
            output = model(images)
            # 计算loss
            loss = criterion(output, target)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度优化
            optimizer.step()

        # 一轮验证一次模型
        val(model, device, test_loader, criterion)

        # 模型保存,一轮保存一次
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'checkpoint_epoch_{}.pth'.format(epoch)))
        print("model saved success")


if __name__ == '__main__':
    # argparse模块，大家可以单独搜索下怎么用，实际效果就是：当字典一样用，方便传参
    # 此处，先把参数预设好

    # 实例化一个参数对象
    parser = argparse.ArgumentParser(description="---------------- 图像分类Sample -----------------")
    # 下面开始正式的加载参数：别名key，及对应的值value
    parser.add_argument('--train-data', default='./data/train/', dest='train_data', help='location of train data')
    parser.add_argument('--test-data', default='./data/test/', dest='test_data', help='location of test data')
    parser.add_argument('--image-size', default=224, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int, help='batch size')

    parser.add_argument('--workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    # 模型相关
    # parser.add_argument('--arch', default='efficientnet-b0', help='arch type of EfficientNet')
    # parser.add_argument('--pretrained', default=True, help='learning rate')
    # parser.add_argument('--advprop', default=False, help='advprop')
    parser.add_argument('--classes_num', default=2, dest='classes_num', help='classes_num')
    # args = parser.parse_args()
    args = parser.parse_args(args=[])

    # 调用主函数
    main(args)