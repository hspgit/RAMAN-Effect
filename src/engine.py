import torch


import numpy as np

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, phase="Train", mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        if mixup_alpha > 0.0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)
            index = torch.randperm(data.size(0)).to(device)
            data = lam * data + (1 - lam) * data[index, :]
            target_a, target_b = target, target[index]
        else:
            lam = 1.0
            target_a, target_b = target, target

        optimizer.zero_grad()
        output = model(data)
        
        if mixup_alpha > 0.0:
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        else:
            loss = criterion(output, target)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        
        if mixup_alpha > 0.0:
            # Track accuracy against the dominant class
            correct += predicted.eq(target_a).sum().item()
        else:
            correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'{phase} Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                  f'Acc: {100. * correct / total:.2f}%')

def evaluate(model, device, data_loader, criterion, phase="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)
    print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
          f'({acc:.2f}%)\n')
    return test_loss, acc
