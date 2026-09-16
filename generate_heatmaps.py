import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import RamanDataset
from src.engine import train_epoch
from src.model import Raman1DCNN, Transformer1D, MultiscaleCNN, FusionNet
from legacy_baseline.model import LegacyCNN

def get_predictions(model, device, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return np.array(all_preds), np.array(all_targets)

def main():
    data_dir = 'data'
    results_dir = 'results/heatmaps'
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Get global label mapping from reference data
    y_ref_path = os.path.join(data_dir, 'y_reference.npy')
    raw_y = np.load(y_ref_path)
    unique_labels = sorted(np.unique(raw_y))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} classes.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Datasets
    print("Loading datasets...")
    train_dataset = RamanDataset(os.path.join(data_dir, 'X_finetune.npy'),
                                 os.path.join(data_dir, 'y_finetune.npy'),
                                 label_mapping=label_mapping)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = RamanDataset(os.path.join(data_dir, 'X_test.npy'),
                                os.path.join(data_dir, 'y_test.npy'),
                                label_mapping=label_mapping)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Generate ordered class names for the axes
    # The dictionary maps original_label -> index. We want the original labels ordered by index.
    idx_to_label = {v: k for k, v in label_mapping.items()}
    class_names = [str(idx_to_label[i]) for i in range(num_classes)]
    
    models_to_test = {
        'legacy_cnn': LegacyCNN,
        'resnet': Raman1DCNN,
        'transformer': Transformer1D,
        'multiscale_cnn': MultiscaleCNN,
        'fusion': FusionNet
    }
    
    epochs = 5
    
    for model_name, model_cls in models_to_test.items():
        print(f"\n--- Processing Model: {model_name} ---")
        model = model_cls(num_classes=num_classes).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few epochs
        print("Training...")
        for epoch in range(1, epochs + 1):
            train_epoch(model, device, train_loader, optimizer, criterion, epoch, phase="Train", mixup_alpha=0.0)
            
        # Get predictions
        print("Evaluating and computing confusion matrix...")
        preds, targets = get_predictions(model, device, test_loader)
        
        cm = confusion_matrix(targets, preds, labels=np.arange(num_classes))
        
        # Plot and save heatmap
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
        
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved heatmap for {model_name} to {save_path}")

if __name__ == '__main__':
    main()
