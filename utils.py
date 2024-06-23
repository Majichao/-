# # # import torch
# # #
# # # def train(model, train_loader, criterion, optimizer, epoch):
# # #     model.train()
# # #     for batch_idx, (data, target) in enumerate(train_loader):
# # #         optimizer.zero_grad()
# # #         output = model(data)
# # #         loss = criterion(output, target)
# # #         loss.backward()
# # #         optimizer.step()
# # #         if batch_idx % 10 == 0:
# # #             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
# # #
# # # def test(model, test_loader, criterion):
# # #     model.eval()
# # #     test_loss = 0
# # #     correct = 0
# # #     with torch.no_grad():
# # #         for data, target in test_loader:
# # #             output = model(data)
# # #             test_loss += criterion(output, target).item()
# # #             pred = output.argmax(dim=1, keepdim=True)
# # #             correct += pred.eq(target.view_as(pred)).sum().item()
# # #
# # #     test_loss /= len(test_loader.dataset)
# # #     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
# # #
# # #
# # #
# #
# #
# # import torch
# #
# # def train(model, train_loader, criterion, optimizer, epoch, device):
# #     model.train()
# #     for batch_idx, (data, target) in enumerate(train_loader):
# #         data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
# #         optimizer.zero_grad()
# #         output = model(data)
# #         loss = criterion(output, target)
# #         loss.backward()
# #         optimizer.step()
# #         if batch_idx % 10 == 0:
# #             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
# #
# # def test(model, test_loader, criterion, device):
# #     model.eval()
# #     test_loss = 0
# #     correct = 0
# #     with torch.no_grad():
# #         for data, target in test_loader:
# #             data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
# #             output = model(data)
# #             test_loss += criterion(output, target).item()
# #             pred = output.argmax(dim=1, keepdim=True)
# #             correct += pred.eq(target.view_as(pred)).sum().item()
# #
# #     test_loss /= len(test_loader.dataset)
# #     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
# #
#
# import torch
#
# def train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list):
#     model.train()
#     correct = 0
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#         if batch_idx % 10 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
#     avg_loss = total_loss / len(train_loader)
#     accuracy = correct / len(train_loader.dataset)
#     train_loss_list.append(avg_loss)
#     train_acc_list.append(accuracy)
#     print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}')
#
# def test(model, test_loader, criterion, device, test_loss_list, test_acc_list):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             if batch_idx % 10 == 0:
#                 print(f'Test: [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')
#
#     avg_loss = test_loss / len(test_loader)
#     accuracy = correct / len(test_loader.dataset)
#     test_loss_list.append(avg_loss)
#     test_acc_list.append(accuracy)
#     print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n')

# import torch
#
# def train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list):
#     model.train()
#     correct = 0
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#         if batch_idx % 10 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
#     avg_loss = total_loss / len(train_loader)
#     accuracy = correct / len(train_loader.dataset)
#     train_loss_list.append(avg_loss)
#     train_acc_list.append(accuracy)
#     print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}')
#
# def test(model, test_loader, criterion, device, test_loss_list, test_acc_list):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             if batch_idx % 10 == 0:
#                 print(f'Test: [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')
#
#     avg_loss = test_loss / len(test_loader)
#     accuracy = correct / len(test_loader.dataset)
#     test_loss_list.append(avg_loss)
#     test_acc_list.append(accuracy)
#     print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n')

# import torch
#
# def train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list):
#     model.train()
#     correct = 0
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#         if batch_idx % 10 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
#     avg_loss = total_loss / len(train_loader)
#     accuracy = correct / len(train_loader.dataset)
#     train_loss_list.append(avg_loss)
#     train_acc_list.append(accuracy)
#     print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}')
#
# def test(model, test_loader, criterion, device, test_loss_list, test_acc_list):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             if batch_idx % 10 == 0:
#                 print(f'Test: [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')
#
#     avg_loss = test_loss / len(test_loader)
#     accuracy = correct / len(test_loader.dataset)
#     test_loss_list.append(avg_loss)
#     test_acc_list.append(accuracy)
#     print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n')

# import torch
#
#
# def train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list):
#     model.train()
#     correct = 0
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#
#         # 过滤无效标签
#         valid_idx = target != -1
#         data, target = data[valid_idx], target[valid_idx]
#
#         if len(target) == 0:
#             continue  # 如果没有有效标签，跳过这个批次
#
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         pred = output.argmax(dim=1, keepdim=True)
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
#         if batch_idx % 10 == 0:
#             print(
#                 f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
#     avg_loss = total_loss / len(train_loader)
#     accuracy = correct / len(train_loader.dataset)
#     train_loss_list.append(avg_loss)
#     train_acc_list.append(accuracy)
#     print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}')
#
#
# def test(model, test_loader, criterion, device, test_loss_list, test_acc_list):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#
#             # 过滤无效标签
#             valid_idx = target != -1
#             data, target = data[valid_idx], target[valid_idx]
#
#             if len(target) == 0:
#                 continue  # 如果没有有效标签，跳过这个批次
#
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#             if batch_idx % 10 == 0:
#                 print(
#                     f'Test: [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')
#
#     avg_loss = test_loss / len(test_loader)
#     accuracy = correct / len(test_loader.dataset)
#     test_loss_list.append(avg_loss)
#     test_acc_list.append(accuracy)
#     print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n')

# import torch
#
#
# def train(model, train_loader, criterion, optimizer, epoch, device, train_loss_list, train_acc_list):
#     model.train()
#     correct = 0
#     total_loss = 0
#     total_samples = 0
#
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#
#         if len(target) == 0:
#             continue  # 跳过空批次
#
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * len(target)
#         pred = output.argmax(dim=1, keepdim=True)
#         valid_idx = target != -1
#         correct += pred[valid_idx].eq(target[valid_idx].view_as(pred[valid_idx])).sum().item()
#         total_samples += valid_idx.sum().item()
#
#         if batch_idx % 10 == 0:
#             print(
#                 f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
#
#     avg_loss = total_loss / total_samples
#     accuracy = correct / total_samples
#     train_loss_list.append(avg_loss)
#     train_acc_list.append(accuracy)
#     print(f'Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}')
#
#
# def test(model, test_loader, criterion, device, test_loss_list, test_acc_list):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total_samples = 0
#
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#
#             if len(target) == 0:
#                 continue  # 跳过空批次
#
#             output = model(data)
#             loss = criterion(output, target)
#             test_loss += loss.item() * len(target)
#             pred = output.argmax(dim=1, keepdim=True)
#             valid_idx = target != -1
#             correct += pred[valid_idx].eq(target[valid_idx].view_as(pred[valid_idx])).sum().item()
#             total_samples += valid_idx.sum().item()
#
#             if batch_idx % 10 == 0:
#                 print(
#                     f'Test: [{batch_idx * len(data)}/{len(test_loader.dataset)} ({100. * batch_idx / len(test_loader):.0f}%)]\tLoss: {test_loss:.6f}')
#
#     avg_loss = test_loss / total_samples
#     accuracy = correct / total_samples
#     test_loss_list.append(avg_loss)
#     test_acc_list.append(accuracy)
#     print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n')

import torch

def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据移动到 GPU
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


