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
