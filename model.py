# import torch.nn as nn
# import torch.nn.functional as F
#
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(3 * 64 * 64, 300)
#         self.fc2 = nn.Linear(300, 100)
#         self.fc3 = nn.Linear(100, 10)  # 输出维度设为10
#
#     def forward(self, x):
#         x = x.view(-1, 3 * 64 * 64)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
#
#
# class ComplexModel(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ComplexModel, self).__init__()
#         self.base_model = resnet18(pretrained=True)
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
#
#     def forward(self, x):
#         x = self.base_model(x)
#         return x

# import torch
# import torch.nn as nn
# from torchvision.models import resnet50
#
#
# class ComplexModel(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ComplexModel, self).__init__()
#         self.base_model = resnet50(weights='IMAGENET1K_V1')
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
#
#     def forward(self, x):
#         x = self.base_model(x)
#         return x

# import torch
# import torch.nn as nn
# from torchvision.models import resnet50
#
#
# class ComplexModel(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ComplexModel, self).__init__()
#         # 使用本地下载的预训练模型文件
#         self.base_model = resnet50(weights=None)
#         checkpoint = torch.load('resnet50-0676ba61.pth')
#         self.base_model.load_state_dict(checkpoint)
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
#
#     def forward(self, x):
#         x = self.base_model(x)
#         return x


# 1.0
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(3 * 64 * 64, 300)
#         self.fc2 = nn.Linear(300, 100)
#         self.fc3 = nn.Linear(100, 10)  # 输出维度设为10
#
#     def forward(self, x):
#         x = x.view(-1, 3 * 64 * 64)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 输出维度设为10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


