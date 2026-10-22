import torch


import numpy as np

# 30 reference isolates mapped to 8 empiric antibiotic treatment groups (from bacteria-ID/config.py)
ATCC_GROUPINGS = {
    3: 0, 4: 0, 9: 0, 10: 0, 2: 0, 8: 0, 11: 0, 22: 0,  # Meropenem
    19: 1,                                               # Ciprofloxacin
    12: 2, 13: 2,                                        # TZP
    14: 3, 18: 3, 15: 3, 20: 3, 21: 3, 16: 3, 17: 3,     # Vancomycin
    23: 4, 24: 4,                                        # Ceftriaxone
    26: 5, 27: 5, 28: 5, 29: 5, 25: 5, 6: 5, 7: 5,       # Penicillin
    5: 6,                                                # Daptomycin
    0: 7, 1: 7                                           # Caspofungin
}

ANTIBIOTICS = {
    0: "Meropenem",
    1: "Ciprofloxacin",
    2: "TZP",
    3: "Vancomycin",
    4: "Ceftriaxone",
    5: "Penicillin",
    6: "Daptomycin",
    7: "Caspofungin"
}

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
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.view(-1).cpu().numpy())

    total = len(data_loader.dataset)
    test_loss /= total
    acc = 100. * correct / total

    # Compute 30-to-8 antibiotic treatment group accuracy if labels are in 0..29
    if all(t in ATCC_GROUPINGS and p in ATCC_GROUPINGS for t, p in zip(all_targets, all_preds)):
        ab_targets = np.array([ATCC_GROUPINGS[int(t)] for t in all_targets])
        ab_preds = np.array([ATCC_GROUPINGS[int(p)] for p in all_preds])
        ab_correct = int((ab_preds == ab_targets).sum())
        ab_acc = 100. * ab_correct / total
        print(f'\n{phase} set: Average loss: {test_loss:.4f}, '
              f'Isolate Accuracy: {correct}/{total} ({acc:.2f}%), '
              f'Treatment Group Accuracy: {ab_correct}/{total} ({ab_acc:.2f}%)\n')
    else:
        print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} '
              f'({acc:.2f}%)\n')
    return test_loss, acc
