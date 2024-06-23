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


