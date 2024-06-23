# import os
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# import torch
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'train2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
#         transform=transform
#     )
#     test_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'val2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_val2017.json'),
#         transform=transform
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = list(zip(*batch))
#     images = torch.stack(images, dim=0)
#
#     # 提取每个目标中的类别标签
#     labels = []
#     for target in targets:
#         if len(target) > 0:
#             labels.append(target[0]['category_id'])
#         else:
#             labels.append(0)  # 若目标为空，附加一个默认类别，例如类别0
#
#     # 过滤掉不在前10个类别范围内的样本
#     filtered_labels = [label if label < 10 else 0 for label in labels]
#     labels = torch.tensor(filtered_labels)
#     return images, labels
#
# import os
# import torch
# from torchvision import datasets
# from torchvision.datasets.coco import CocoDetection
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dir = os.path.join(data_dir, 'train2017')
#     val_dir = os.path.join(data_dir, 'val2017')
#     train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#     val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#
#     train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
#     val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                collate_fn=collate_fn)
#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack([img for img in images], dim=0)
#
#     labels = []
#     for target in targets:
#         if len(target) > 0:
#             labels.append(target[0]['category_id'])
#         else:
#             labels.append(-1)  # 添加一个默认标签，避免索引错误
#
#     labels = torch.tensor(labels)
#
#     # 调试代码：打印标签的最小值和最大值
#     print(f"Labels: min={labels.min()}, max={labels.max()}")
#
#     # 确保标签在有效范围内
#     labels = torch.clamp(labels, min=0, max=9)
#
#     return images, labels



# import os
# import torch
# from torchvision import datasets
# from torchvision.datasets.coco import CocoDetection
#
#
# # 重新映射标签函数
# def remap_labels(targets, max_classes=10):
#     label_mapping = {}
#     remapped_labels = []
#     current_label = 0
#
#     for target in targets:
#         if len(target) > 0:
#             original_label = target[0]['category_id']
#             if original_label not in label_mapping:
#                 if current_label < max_classes:
#                     label_mapping[original_label] = current_label
#                     current_label += 1
#                 else:
#                     label_mapping[original_label] = -1
#             remapped_labels.append(label_mapping[original_label])
#         else:
#             remapped_labels.append(-1)
#
#     return torch.tensor(remapped_labels), label_mapping
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dir = os.path.join(data_dir, 'train2017')
#     val_dir = os.path.join(data_dir, 'val2017')
#     train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#     val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#
#     train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
#     val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                collate_fn=collate_fn)
#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack([img for img in images], dim=0)
#
#     labels, label_mapping = remap_labels(targets)
#
#     # 调试代码：打印标签的最小值和最大值以及映射表
#     print(f"Labels: min={labels.min()}, max={labels.max()}")
#     print(f"Label mapping: {label_mapping}")
#
#     # 确保标签在有效范围内
#     labels = torch.clamp(labels, min=0, max=9)
#
#     return images, labels

# import os
# import torch
# from torchvision import datasets
# from torchvision.datasets.coco import CocoDetection
#
#
# def remap_labels(targets, max_classes=10):
#     label_mapping = {}
#     remapped_labels = []
#     current_label = 0
#
#     for target in targets:
#         if len(target) > 0:
#             original_label = target[0]['category_id']
#             if original_label not in label_mapping:
#                 if current_label < max_classes:
#                     label_mapping[original_label] = current_label
#                     current_label += 1
#                 else:
#                     label_mapping[original_label] = -1
#             remapped_labels.append(label_mapping[original_label])
#         else:
#             remapped_labels.append(-1)
#
#     return torch.tensor(remapped_labels), label_mapping
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dir = os.path.join(data_dir, 'train2017')
#     val_dir = os.path.join(data_dir, 'val2017')
#     train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#     val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#
#     train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
#     val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                collate_fn=collate_fn)
#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack([img for img in images], dim=0)
#
#     labels, label_mapping = remap_labels(targets)
#
#     # 调试代码：打印标签的最小值和最大值以及映射表
#     print(f"Labels: min={labels.min()}, max={labels.max()}")
#     print(f"Label mapping: {label_mapping}")
#
#     # 确保标签在有效范围内
#     labels = torch.clamp(labels, min=0, max=9)
#
#     return images, labels

# import os
# import torch
# from torchvision import datasets
# from torchvision.datasets.coco import CocoDetection
#
#
# def remap_labels(targets, max_classes=10):
#     label_mapping = {}
#     remapped_labels = []
#     current_label = 0
#
#     for target in targets:
#         if len(target) > 0:
#             original_label = target[0]['category_id']
#             if original_label not in label_mapping:
#                 if current_label < max_classes:
#                     label_mapping[original_label] = current_label
#                     current_label += 1
#                 else:
#                     label_mapping[original_label] = -1  # 超出范围的标签设为 -1
#             remapped_labels.append(label_mapping[original_label])
#         else:
#             remapped_labels.append(-1)  # 添加一个默认标签，避免索引错误
#
#     # 去除 -1 的标签并保留有效标签
#     valid_labels = [label for label in remapped_labels if label != -1]
#
#     return torch.tensor(valid_labels), label_mapping
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dir = os.path.join(data_dir, 'train2017')
#     val_dir = os.path.join(data_dir, 'val2017')
#     train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#     val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#
#     train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
#     val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                collate_fn=collate_fn)
#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack([img for img in images], dim=0)
#
#     labels, label_mapping = remap_labels(targets)
#
#     # 如果过滤后的标签数量少于图片数量，用 -1 填充
#     if len(labels) < len(images):
#         fill_labels = torch.full((len(images) - len(labels),), -1, dtype=torch.long)
#         labels = torch.cat((labels, fill_labels))
#
#     # 调试代码：打印标签的最小值和最大值以及映射表
#     print(f"Labels: min={labels.min() if len(labels) > 0 else 'N/A'}, max={labels.max() if len(labels) > 0 else 'N/A'}")
#     print(f"Label mapping: {label_mapping}")
#
#     return images, labels

# import os
# import torch
# from torchvision import datasets
# from torchvision.datasets.coco import CocoDetection
#
#
# def remap_labels(labels, max_classes=10):
#     label_mapping = {}
#     remapped_labels = []
#     current_label = 0
#
#     for label in labels:
#         if label not in label_mapping:
#             if current_label < max_classes:
#                 label_mapping[label] = current_label
#                 current_label += 1
#             else:
#                 label_mapping[label] = -1  # 超出范围的标签设为 -1
#         remapped_labels.append(label_mapping[label])
#
#     return torch.tensor(remapped_labels), label_mapping
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dir = os.path.join(data_dir, 'train2017')
#     val_dir = os.path.join(data_dir, 'val2017')
#     train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
#     val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
#
#     train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
#     val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                                collate_fn=collate_fn)
#     test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = zip(*batch)
#
#     filtered_images = []
#     filtered_labels = []
#
#     for img, target in zip(images, targets):
#         if len(target) > 0:
#             label = target[0]['category_id']
#             filtered_images.append(img)
#             filtered_labels.append(label)
#
#     images = torch.stack(filtered_images, dim=0)
#     labels, label_mapping = remap_labels(filtered_labels)
#
#     # 调试代码：打印标签的最小值和最大值以及映射表
#     print(f"Labels: min={labels.min() if len(labels) > 0 else 'N/A'}, max={labels.max() if len(labels) > 0 else 'N/A'}")
#     print(f"Label mapping: {label_mapping}")
#
#     return images, labels

# import os
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# import torch
#
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'train2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
#         transform=transform
#     )
#     test_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'val2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_val2017.json'),
#         transform=transform
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
#
# def collate_fn(batch):
#     images, targets = list(zip(*batch))
#     images = torch.stack(images, dim=0)
#
#     # 提取每个目标中的类别标签
#     labels = []
#     for target in targets:
#         if len(target) > 0:
#             labels.append(target[0]['category_id'])
#         else:
#             labels.append(0)  # 若目标为空，附加一个默认类别，例如类别0
#
#     # 过滤掉不在前10个类别范围内的样本
#     filtered_labels = [label if label < 10 else 0 for label in labels]
#     labels = torch.tensor(filtered_labels)
#     return images, labels

# import os
# from torchvision.datasets import CocoDetection
# from torch.utils.data import DataLoader
# import torch
#
# def get_coco_dataloader(data_dir, batch_size, transform):
#     train_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'train2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
#         transform=transform
#     )
#     test_dataset = CocoDetection(
#         root=os.path.join(data_dir, 'val2017'),
#         annFile=os.path.join(data_dir, 'annotations', 'instances_val2017.json'),
#         transform=transform
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#
#     return train_loader, test_loader
#
# def collate_fn(batch):
#     images, targets = list(zip(*batch))
#     images = torch.stack(images, dim=0)
#     # 提取每个目标中的类别标签
#     labels = [target[0]['category_id'] for target in targets]
#     # 过滤掉不在前10个类别范围内的样本
#     filtered_labels = [label if label < 10 else 0 for label in labels]
#     labels = torch.tensor(filtered_labels)
#     return images, labels

import os
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torch


def get_coco_dataloader(data_dir, batch_size, transform):
    train_dataset = CocoDetection(
        root=os.path.join(data_dir, 'train2017'),
        annFile=os.path.join(data_dir, 'annotations', 'instances_train2017.json'),
        transform=transform
    )
    test_dataset = CocoDetection(
        root=os.path.join(data_dir, 'val2017'),
        annFile=os.path.join(data_dir, 'annotations', 'instances_val2017.json'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)

    # 提取每个目标中的类别标签
    labels = []
    for target in targets:
        if len(target) > 0:
            labels.append(target[0]['category_id'])
        else:
            labels.append(0)  # 若目标为空，附加一个默认类别，例如类别0

    # 过滤掉不在前10个类别范围内的样本
    filtered_labels = [label if label < 10 else 0 for label in labels]
    labels = torch.tensor(filtered_labels)
    return images, labels


