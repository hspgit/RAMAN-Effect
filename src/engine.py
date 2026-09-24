import numpy as np
import torch

from src.metrics import confusion, format_summary, summarize


def train_epoch(model, device, train_loader, optimizer, criterion, epoch,
                phase="Train", mixup_alpha=0.0):
    """Train one epoch.

    Behaviour is unchanged. The only difference is the return value: previously
    None, now a dict of epoch statistics so the caller can record them. The
    printed lines are byte-identical to before so webapp/app.py's regex keeps
    working.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

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
        n_batches += 1
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

    return {
        "train_loss": running_loss / max(n_batches, 1),
        "train_acc": 100.0 * correct / max(total, 1),
        "n_batches": n_batches,
        "n_samples": total,
    }


def evaluate(model, device, data_loader, criterion, phase="Test",
             num_classes=None, return_stats=False):
    """Evaluate and report per-class metrics, not just one accuracy scalar.

    Returns (loss, acc) as before. Pass return_stats=True for
    (loss, acc, stats) where stats carries macro-F1, per-class recall and
    precision, support, and the full confusion matrix.

    Why this matters: the characteristic failure of retraining on new data is
    per-class. Overall accuracy can hold steady while several classes collapse.
    A single scalar cannot show that.

    Pass num_classes explicitly. Inferring it from the labels present breaks
    the moment you evaluate on a subset of classes.
    """
    model.eval()
    test_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            preds.append(output.argmax(dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    n = len(data_loader.dataset)
    test_loss /= n
    correct = int(np.sum(y_true == y_pred))
    acc = 100.0 * correct / n

    k = num_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion(y_true, y_pred, k)
    stats = summarize(cm)
    stats["loss"] = test_loss
    stats["confusion"] = cm.tolist()
    stats["y_pred"] = y_pred
    stats["y_true"] = y_true

    # First line kept byte-identical for webapp/app.py compatibility.
    print(f'\n{phase} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{n} '
          f'({acc:.2f}%)\n')
    print(format_summary(stats, phase) + "\n")

    if return_stats:
        return test_loss, acc, stats
    return test_loss, acc
