import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model import SimpleModel
from prune import prune_model
from utils import train, test
from data import get_coco_dataloader


def main():
    # 参数设置
    batch_size = 32
    epochs = 10
    lr = 0.0001  # 学习率
    pruning_rate = 0.2

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据
    train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)

    # 定义模型
    model = SimpleModel().to(device)  # 将模型移动到 GPU
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用 Adam 优化器

    # 训练模型
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        test(model, test_loader, criterion, device)

    # 剪枝模型
    prune_model(model, pruning_rate)

    # 训练剪枝后的模型
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        test(model, test_loader, criterion, device)


if __name__ == '__main__':
    main()




