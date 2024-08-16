# 单张图片分类可视化
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model import buid_model

"""
单张图片预测，可视化脚本
"""


def main(img_path, model_path):
    """
    读取图片，加载模型，预测可视化结果
    :param img_path:
    :param model_path:
    :return:
    """
    # 读取图像
    img = Image.open(img_path)
    img1 = img.copy()
    # 图片预处理，保证和训练是一样
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 图片缩放到（224，224）
        transforms.ToTensor(),  # 将图像数据格式转换为Tensor格式。所有数除以255，将数据归一化到[0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准正态分布变换，三通道归一化到[-1,1]。(input-mean)/std
    ])
    image = transform(img)  # (3, 224, 224)
    image = image.unsqueeze(0)  # 增加batch维度  (1, 3, 224, 224)
    # print(image.shape)

    # 模型加载
    model = buid_model(2)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()  # 必须是eval模式，否则模型预测不准

    # 预测
    out = model(image)  # out: (1, 2)
    pro = F.softmax(out, 1)  # 结果进行softmax,得到每个类别的预测概率 pro: (1, 2)
    _, pred = torch.max(out.data, 1)  # 获取模型输出结果最大值索引，得到预测类别
    cls_index = int(pred)  # 类别索引
    cls_pro = round(float(pro[0][cls_index]), 2)  # 类别概率，保留两位小数
    print(cls_index, cls_pro)

    # 可视化模型分类结果
    # 创建一个可以在给定图片上绘图的对象
    cl_dic = {0: "Cat", 1: "Dog"}
    draw = ImageDraw.Draw(img1)
    # 定义字体和大小
    font = ImageFont.truetype("arial.ttf", 10)
    # 写字
    draw.text((10, 10), "class:" + cl_dic[cls_index]+"   pro:"+str(cls_pro), font=font, fill=(0, 255, 255))
    # 保存新的图片
    img1.show()


if __name__ == '__main__':
    img_path = "./data/test/Cat/5.jpg"  # 图片路径
    # img_path = "./data/test/Dog/5.jpg"
    model_path = "./ckpts/checkpoint_epoch_3.pth"  # 模型权重路径
    main(img_path, model_path)
