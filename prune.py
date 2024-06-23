# # import torch
# # import torch.nn.utils.prune as prune
# #
# # def prune_model(model, pruning_rate):
# #     parameters_to_prune = []
# #     for name, module in model.named_modules():
# #         if isinstance(module, torch.nn.Linear):
# #             parameters_to_prune.append((module, 'weight'))
# #
# #     prune.global_unstructured(
# #         parameters_to_prune,
# #         pruning_method=prune.L1Unstructured,
# #         amount=pruning_rate,
# #     )
# #
# #     # 移除剪枝后的参数
# #     for module, param in parameters_to_prune:
# #         prune.remove(module, param)
# #
# #     # 查看剪枝后的稀疏率
# #     for module, param in parameters_to_prune:
# #         print(f'Sparsity in {module} {param}: {100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())}%')
#
# import torch
# import torch.nn.utils.prune as prune
#
# def prune_model_stepwise(model, pruning_rate, train_loader, test_loader, criterion, optimizer, device, epochs=5):
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             print(f'Pruning {name}...')
#             prune.l1_unstructured(module, name='weight', amount=pruning_rate)
#             print(f'Sparsity in {name}: {100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()):.2f}%')
#
#             # 重新训练剪枝后的层
#             for epoch in range(epochs):
#                 train(model, train_loader, criterion, optimizer, epoch, device, [], [])
#                 test(model, test_loader, criterion, device, [], [])
#
#             prune.remove(module, 'weight')
#             print(f'Finished pruning {name}.')
#
# def prune_model_global(model, pruning_rate):
#     parameters_to_prune = []
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             parameters_to_prune.append((module, 'weight'))
#
#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=pruning_rate,
#     )
#
#     for module, param in parameters_to_prune:
#         prune.remove(module, param)
#
#     for module, param in parameters_to_prune:
#         print(f'Sparsity in {module} {param}: {100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())}%')

# import torch
# import torch.nn as nn
# import torch.nn.utils.prune as prune
#
# def compute_importance_scores(model, dataloader, criterion, device):
#     model.eval()
#     scores = {}
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             scores[name] = torch.zeros_like(module.weight, device=device)
#
#     for data, target in dataloader:
#         data, target = data.to(device), target.to(device)
#         model.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Linear):
#                 scores[name] += module.weight.grad.abs()
#
#     for name in scores:
#         scores[name] /= len(dataloader.dataset)
#
#     return scores
#
# def prune_model_gradually(model, scores, pruning_rate, device):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.custom_from_mask(module, 'weight', mask=scores[name] > torch.quantile(scores[name], pruning_rate))
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.remove(module, 'weight')
#
# def prune_and_retrain(model, dataloader, criterion, optimizer, device, pruning_rate, retrain_epochs, total_pruning_steps):
#     for step in range(total_pruning_steps):
#         print(f"Pruning step {step + 1}/{total_pruning_steps}...")
#         scores = compute_importance_scores(model, dataloader, criterion, device)
#         prune_model_gradually(model, scores, pruning_rate, device)
#
#         for epoch in range(retrain_epochs):
#             train(model, dataloader, criterion, optimizer, epoch, device, [], [])
#             test(model, dataloader, criterion, device, [], [])

# import torch
# import torch.nn as nn
# import torch.nn.utils.prune as prune
#
# def compute_importance_scores(model, dataloader, criterion, device):
#     model.eval()
#     scores = {}
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             scores[name] = torch.zeros_like(module.weight, device=device)
#
#     for data, target in dataloader:
#         data, target = data.to(device), target.to(device)
#         model.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Linear):
#                 scores[name] += module.weight.grad.abs()
#
#     for name in scores:
#         scores[name] /= len(dataloader.dataset)
#
#     return scores
#
# def prune_model_gradually(model, scores, pruning_rate, device):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.custom_from_mask(module, 'weight', mask=scores[name] > torch.quantile(scores[name], pruning_rate))
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.remove(module, 'weight')
#
# def prune_and_retrain(model, dataloader, criterion, optimizer, device, pruning_rate, retrain_epochs, total_pruning_steps):
#     for step in range(total_pruning_steps):
#         print(f"Pruning step {step + 1}/{total_pruning_steps}...")
#         scores = compute_importance_scores(model, dataloader, criterion, device)
#         prune_model_gradually(model, scores, pruning_rate, device)
#
#         for epoch in range(retrain_epochs):
#             train(model, dataloader, criterion, optimizer, epoch, device, [], [])
#             test(model, dataloader, criterion, device, [], [])

# import torch
# import torch.nn as nn
# import torch.nn.utils.prune as prune
#
# def compute_importance_scores(model, dataloader, criterion, device):
#     model.eval()
#     scores = {}
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             scores[name] = torch.zeros_like(module.weight, device=device)
#
#     for data, target in dataloader:
#         data, target = data.to(device), target.to(device)
#         model.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Linear):
#                 scores[name] += module.weight.grad.abs()
#
#     for name in scores:
#         scores[name] /= len(dataloader.dataset)
#
#     return scores
#
# def prune_model_gradually(model, scores, pruning_rate, device):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.custom_from_mask(module, 'weight', mask=scores[name] > torch.quantile(scores[name], pruning_rate))
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             prune.remove(module, 'weight')
#
# def prune_and_retrain(model, dataloader, criterion, optimizer, device, pruning_rate, retrain_epochs, total_pruning_steps):
#     for step in range(total_pruning_steps):
#         print(f"Pruning step {step + 1}/{total_pruning_steps}...")
#         scores = compute_importance_scores(model, dataloader, criterion, device)
#         prune_model_gradually(model, scores, pruning_rate, device)
#
#         for epoch in range(retrain_epochs):
#             train(model, dataloader, criterion, optimizer, epoch, device, [], [])
#             test(model, dataloader, criterion, device, [], [])

import torch
import torch.nn.utils.prune as prune

def prune_model(model, pruning_rate):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

    # 移除剪枝后的参数
    for module, param in parameters_to_prune:
        prune.remove(module, param)

    # 查看剪枝后的稀疏率
    for module, param in parameters_to_prune:
        print(f'Sparsity in {module} {param}: {100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())}%')
