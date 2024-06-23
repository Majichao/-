# # # # # import torch
# # # # # import torch.nn as nn
# # # # # import torch.optim as optim
# # # # # from torchvision import transforms
# # # # # from model import SimpleModel
# # # # # from prune import prune_model
# # # # # from utils import train, test
# # # # # from data import get_coco_dataloader
# # # # #
# # # # #
# # # # # def main():
# # # # #     # 参数设置
# # # # #     batch_size = 16
# # # # #     epochs = 10
# # # # #     lr = 0.01
# # # # #     pruning_rate = 0.2
# # # # #
# # # # #     # 数据预处理
# # # # #     transform = transforms.Compose([
# # # # #         transforms.Resize((64, 64)),
# # # # #         transforms.ToTensor(),
# # # # #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # # # #     ])
# # # # #
# # # # #     # 加载数据
# # # # #     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
# # # # #
# # # # #     # 定义模型
# # # # #     model = SimpleModel()
# # # # #     criterion = nn.CrossEntropyLoss()
# # # # #     optimizer = optim.SGD(model.parameters(), lr=lr)
# # # # #
# # # # #     # 训练模型
# # # # #     for epoch in range(1, epochs + 1):
# # # # #         train(model, train_loader, criterion, optimizer, epoch)
# # # # #         test(model, test_loader, criterion)
# # # # #
# # # # #     # 剪枝模型
# # # # #     prune_model(model, pruning_rate)
# # # # #
# # # # #     # 训练剪枝后的模型
# # # # #     for epoch in range(1, epochs + 1):
# # # # #         train(model, train_loader, criterion, optimizer, epoch)
# # # # #         test(model, test_loader, criterion)
# # # # #
# # # # #
# # # # # if __name__ == '__main__':
# # # # #     main()
# # # #
# # # #
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # from torchvision import transforms
# # # # from model import SimpleModel
# # # # from prune import prune_model
# # # # from utils import train, test
# # # # from data import get_coco_dataloader
# # # #
# # # #
# # # # def main():
# # # #     # 参数设置
# # # #     batch_size = 16
# # # #     epochs = 10
# # # #     lr = 0.01
# # # #     pruning_rate = 0.2
# # # #
# # # #     # 检查是否有可用的 GPU
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     print("Using device:", device)
# # # #
# # # #     # 数据预处理
# # # #     transform = transforms.Compose([
# # # #         transforms.Resize((64, 64)),
# # # #         transforms.ToTensor(),
# # # #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # # #     ])
# # # #
# # # #     # 加载数据
# # # #     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
# # # #
# # # #     # 定义模型
# # # #     model = SimpleModel().to(device)  # 将模型移动到 GPU
# # # #     criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
# # # #     optimizer = optim.SGD(model.parameters(), lr=lr)
# # # #
# # # #     # 训练模型
# # # #     for epoch in range(1, epochs + 1):
# # # #         train(model, train_loader, criterion, optimizer, epoch, device)
# # # #         test(model, test_loader, criterion, device)
# # # #
# # # #     # 剪枝模型
# # # #     prune_model(model, pruning_rate)
# # # #
# # # #     # 训练剪枝后的模型
# # # #     for epoch in range(1, epochs + 1):
# # # #         train(model, train_loader, criterion, optimizer, epoch, device)
# # # #         test(model, test_loader, criterion, device)
# # # #
# # # #
# # # # if __name__ == '__main__':
# # # #     main()
# # # #
# # # #
# # # #
# # # #
# # #
# # #
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torchvision import transforms
# # # from model import SimpleModel
# # # from prune import prune_model
# # # from utils import train, test
# # # from data import get_coco_dataloader
# # #
# # #
# # # def main():
# # #     # 参数设置
# # #     batch_size = 16
# # #     epochs = 10
# # #     lr = 0.01
# # #     pruning_rate = 0.2
# # #
# # #     # 检查是否有可用的 GPU
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     print("Using device:", device)
# # #
# # #     # 数据预处理
# # #     transform = transforms.Compose([
# # #         transforms.Resize((64, 64)),
# # #         transforms.ToTensor(),
# # #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # #     ])
# # #
# # #     # 加载数据
# # #     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
# # #
# # #     # 定义模型
# # #     model = SimpleModel().to(device)  # 将模型移动到 GPU
# # #     criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
# # #     optimizer = optim.SGD(model.parameters(), lr=lr)
# # #
# # #     # 训练模型
# # #     for epoch in range(1, epochs + 1):
# # #         train(model, train_loader, criterion, optimizer, epoch, device)
# # #         test(model, test_loader, criterion, device)
# # #
# # #     # 剪枝模型
# # #     prune_model(model, pruning_rate)
# # #
# # #     # 训练剪枝后的模型
# # #     for epoch in range(1, epochs + 1):
# # #         train(model, train_loader, criterion, optimizer, epoch, device)
# # #         test(model, test_loader, criterion, device)
# # #
# # #
# # # if __name__ == '__main__':
# # #     main()
# # #
# # #
# # #
# #
# #
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torchvision import transforms
# # from model import SimpleModel
# # from prune import prune_model
# # from utils import train, test
# # from data import get_coco_dataloader
# # import matplotlib.pyplot as plt
# # import time
# # import os
# #
# #
# # def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
# #     epochs = range(1, len(train_loss_list) + 1)
# #
# #     plt.figure(figsize=(12, 6))
# #
# #     # Plotting loss
# #     plt.subplot(1, 2, 1)
# #     plt.plot(epochs, train_loss_list, label='Train Loss')
# #     plt.plot(epochs, test_loss_list, label='Test Loss')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Loss')
# #     plt.title(f'Training and Testing Loss {title_suffix}')
# #     plt.legend()
# #
# #     # Plotting accuracy
# #     plt.subplot(1, 2, 2)
# #     plt.plot(epochs, train_acc_list, label='Train Accuracy')
# #     plt.plot(epochs, test_acc_list, label='Test Accuracy')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Accuracy')
# #     plt.title(f'Training and Testing Accuracy {title_suffix}')
# #     plt.legend()
# #
# #     plt.tight_layout()
# #     plt.show()
# #
# #
# # def main():
# #     # 参数设置
# #     batch_size = 16
# #     epochs = 10
# #     lr = 0.01
# #     pruning_rate = 0.2
# #
# #     # 检查是否有可用的 GPU
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print("Using device:", device)
# #
# #     # 数据预处理
# #     transform = transforms.Compose([
# #         transforms.Resize((64, 64)),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #     ])
# #
# #     # 加载数据
# #     print("Loading data...")
# #     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
# #     print("Data loaded.")
# #
# #     # 定义模型
# #     model = SimpleModel().to(device)
# #     criterion = nn.CrossEntropyLoss().to(device)
# #     optimizer = optim.SGD(model.parameters(), lr=lr)
# #
# #     # 记录损失和准确率
# #     train_loss_list, train_acc_list = [], []
# #     test_loss_list, test_acc_list = [], []
# #
# #     # 剪枝前训练和测试
# #     print("Training before pruning:")
# #     for epoch in range(1, epochs + 1):
# #         print(f"Epoch {epoch} start.")
# #         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
# #         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
# #         print(f"Epoch {epoch} end.")
# #
# #     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
# #
# #     # 记录剪枝前的模型大小
# #     torch.save(model.state_dict(), "model_before_pruning.pth")
# #     model_size_before = os.path.getsize("model_before_pruning.pth")
# #     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
# #
# #     # 记录剪枝前的推理时间
# #     start_time = time.time()
# #     test(model, test_loader, criterion, device, [], [])
# #     inference_time_before = time.time() - start_time
# #     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
# #
# #     # 剪枝模型
# #     print(f'Pruning model with pruning rate: {pruning_rate}')
# #     model = prune_model(model, pruning_rate)
# #
# #     # 记录剪枝后的模型大小
# #     torch.save(model.state_dict(), "model_after_pruning.pth")
# #     model_size_after = os.path.getsize("model_after_pruning.pth")
# #     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
# #
# #     # 记录剪枝后的推理时间
# #     start_time = time.time()
# #     test(model, test_loader, criterion, device, [], [])
# #     inference_time_after = time.time() - start_time
# #     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
# #
# #     # 重新初始化记录列表
# #     train_loss_list, train_acc_list = [], []
# #     test_loss_list, test_acc_list = [], []
# #
# #     # 剪枝后重新训练和测试
# #     print("Training after pruning:")
# #     for epoch in range(1, epochs + 1):
# #         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
# #         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
# #
# #     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
# #
# #     # 可视化模型大小和推理时间
# #     labels = ['Before Pruning', 'After Pruning']
# #     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
# #     inference_times = [inference_time_before, inference_time_after]
# #
# #     plt.figure(figsize=(12, 6))
# #
# #     # 模型大小
# #     plt.subplot(1, 2, 1)
# #     plt.bar(labels, model_sizes, color=['blue', 'orange'])
# #     plt.ylabel('Model Size (MB)')
# #     plt.title('Model Size Comparison')
# #
# #     # 推理时间
# #     plt.subplot(1, 2, 2)
# #     plt.bar(labels, inference_times, color=['blue', 'orange'])
# #     plt.ylabel('Inference Time (seconds)')
# #     plt.title('Inference Time Comparison')
# #
# #     plt.tight_layout()
# #     plt.show()
# #
# #
# # if __name__ == '__main__':
# #     main()
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import SimpleModel
# from prune import prune_model_stepwise
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
# import os
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.01
#     pruning_rate = 0.2
#     retrain_epochs = 5  # 剪枝后每一层重新训练的轮数
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = SimpleModel().to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_model_stepwise(model, pruning_rate, train_loader, test_loader, criterion, optimizer, device, retrain_epochs)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import ComplexModel
# from prune import prune_and_retrain
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
# import os
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.001
#     pruning_rate = 0.2
#     retrain_epochs = 5
#     total_pruning_steps = 5  # 总的剪枝步数，每次剪枝后重新训练
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = ComplexModel(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         scheduler.step()
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝并重新训练模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_and_retrain(model, train_loader, criterion, optimizer, device, pruning_rate, retrain_epochs,
#                       total_pruning_steps)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import SimpleModel
# from prune import prune_and_retrain
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
# import os
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.01
#     pruning_rate = 0.2
#     retrain_epochs = 5
#     total_pruning_steps = 5  # 总的剪枝步数，每次剪枝后重新训练
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = SimpleModel().to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝并重新训练模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_and_retrain(model, train_loader, criterion, optimizer, device, pruning_rate, retrain_epochs,
#                       total_pruning_steps)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import ComplexModel
# from prune import prune_and_retrain
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.001
#     pruning_rate = 0.2
#     retrain_epochs = 5
#     total_pruning_steps = 5  # 总的剪枝步数，每次剪枝后重新训练
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = ComplexModel(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         scheduler.step()
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝并重新训练模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_and_retrain(model, train_loader, criterion, optimizer, device, pruning_rate, retrain_epochs,
#                       total_pruning_steps)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import ComplexModel
# from prune import prune_and_retrain
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.001
#     pruning_rate = 0.2
#     retrain_epochs = 5
#     total_pruning_steps = 5  # 总的剪枝步数，每次剪枝后重新训练
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = ComplexModel(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         scheduler.step()
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝并重新训练模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_and_retrain(model, train_loader, criterion, optimizer, device, pruning_rate, retrain_epochs,
#                       total_pruning_steps)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 保存最终模型
#     torch.save(model.state_dict(), "final_model.pth")
#     print("Final model saved.")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import ComplexModel
# from prune import prune_and_retrain
# from utils import train, test
# from data import get_coco_dataloader
# import matplotlib.pyplot as plt
# import time
#
#
# def plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, title_suffix):
#     epochs = range(1, len(train_loss_list) + 1)
#
#     plt.figure(figsize=(12, 6))
#
#     # Plotting loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss_list, label='Train Loss')
#     plt.plot(epochs, test_loss_list, label='Test Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Testing Loss {title_suffix}')
#     plt.legend()
#
#     # Plotting accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc_list, label='Train Accuracy')
#     plt.plot(epochs, test_acc_list, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Testing Accuracy {title_suffix}')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 10
#     lr = 0.001
#     pruning_rate = 0.2
#     retrain_epochs = 5
#     total_pruning_steps = 5  # 总的剪枝步数，每次剪枝后重新训练
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     print("Loading data...")
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#     print("Data loaded.")
#
#     # 定义模型
#     model = ComplexModel(num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)  # 设置 ignore_index=-1
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#
#     # 记录损失和准确率
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝前训练和测试
#     print("Training before pruning:")
#     for epoch in range(1, epochs + 1):
#         print(f"Epoch {epoch} start.")
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#         scheduler.step()
#         print(f"Epoch {epoch} end.")
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(Before Pruning)")
#
#     # 记录剪枝前的模型大小
#     torch.save(model.state_dict(), "model_before_pruning.pth")
#     model_size_before = os.path.getsize("model_before_pruning.pth")
#     print(f"Model size before pruning: {model_size_before / 1e6:.2f} MB")
#
#     # 记录剪枝前的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_before = time.time() - start_time
#     print(f"Inference time before pruning: {inference_time_before:.2f} seconds")
#
#     # 剪枝并重新训练模型
#     print(f'Pruning model with pruning rate: {pruning_rate}')
#     prune_and_retrain(model, train_loader, criterion, optimizer, device, pruning_rate, retrain_epochs,
#                       total_pruning_steps)
#
#     # 记录剪枝后的模型大小
#     torch.save(model.state_dict(), "model_after_pruning.pth")
#     model_size_after = os.path.getsize("model_after_pruning.pth")
#     print(f"Model size after pruning: {model_size_after / 1e6:.2f} MB")
#
#     # 记录剪枝后的推理时间
#     start_time = time.time()
#     test(model, test_loader, criterion, device, [], [])
#     inference_time_after = time.time() - start_time
#     print(f"Inference time after pruning: {inference_time_after:.2f} seconds")
#
#     # 重新初始化记录列表
#     train_loss_list, train_acc_list = [], []
#     test_loss_list, test_acc_list = [], []
#
#     # 剪枝后重新训练和测试
#     print("Training after pruning:")
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list)
#         test(model, test_loader, criterion, device, test_loss_list, test_acc_list)
#
#     plot_metrics(train_loss_list, train_acc_list, test_loss_list, test_acc_list, "(After Pruning)")
#
#     # 保存最终模型
#     torch.save(model.state_dict(), "final_model.pth")
#     print("Final model saved.")
#
#     # 可视化模型大小和推理时间
#     labels = ['Before Pruning', 'After Pruning']
#     model_sizes = [model_size_before / 1e6, model_size_after / 1e6]
#     inference_times = [inference_time_before, inference_time_after]
#
#     plt.figure(figsize=(12, 6))
#
#     # 模型大小
#     plt.subplot(1, 2, 1)
#     plt.bar(labels, model_sizes, color=['blue', 'orange'])
#     plt.ylabel('Model Size (MB)')
#     plt.title('Model Size Comparison')
#
#     # 推理时间
#     plt.subplot(1, 2, 2)
#     plt.bar(labels, inference_times, color=['blue', 'orange'])
#     plt.ylabel('Inference Time (seconds)')
#     plt.title('Inference Time Comparison')
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

# 1.0
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from model import SimpleModel
# from prune import prune_model
# from utils import train, test
# from data import get_coco_dataloader
#
#
# def main():
#     # 参数设置
#     batch_size = 16
#     epochs = 1
#     lr = 0.0001
#     pruning_rate = 0.2
#
#     # 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # 加载数据
#     train_loader, test_loader = get_coco_dataloader('./coco_dataset', batch_size, transform)
#
#     # 定义模型
#     model = SimpleModel().to(device)  # 将模型移动到 GPU
#     criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到 GPU
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#
#     # 训练模型
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device)
#         test(model, test_loader, criterion, device)
#
#     # 剪枝模型
#     prune_model(model, pruning_rate)
#
#     # 训练剪枝后的模型
#     for epoch in range(1, epochs + 1):
#         train(model, train_loader, criterion, optimizer, epoch, device)
#         test(model, test_loader, criterion, device)
#
#
# if __name__ == '__main__':
#     main()

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
    epochs = 10  # 增加训练轮数
    lr = 0.0001  # 调整学习率
    pruning_rate = 0.2

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),  # 添加随机水平翻转
        transforms.RandomRotation(10),  # 添加随机旋转
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




