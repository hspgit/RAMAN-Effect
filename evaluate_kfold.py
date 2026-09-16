import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.dataset import RamanDataset, AugmentedDataset
from src.engine import train_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description='5-Fold Finetune Evaluation')
    parser.add_argument('--model-type', type=str, default='legacy_cnn')
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Reference Data for Pre-training
    print("\nLoading reference dataset...")
    full_ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                                    os.path.join(args.data_dir, 'y_reference.npy'))
    num_classes = len(np.unique(full_ref_dataset.y))
    
    val_size = int(0.1 * len(full_ref_dataset))
    train_size = len(full_ref_dataset) - val_size
    ref_train, ref_val = random_split(full_ref_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    ref_train_wrapped = AugmentedDataset(ref_train, augment=args.augment)
    ref_val_wrapped = AugmentedDataset(ref_val, augment=False)
    
    ref_train_loader = DataLoader(ref_train_wrapped, batch_size=args.batch_size, shuffle=True)
    ref_val_loader = DataLoader(ref_val_wrapped, batch_size=args.batch_size*2, shuffle=False)

    def get_model():
        if args.model_type == 'legacy_cnn':
            from legacy_baseline.model import LegacyCNN
            return LegacyCNN(num_classes=num_classes).to(device)
        else:
            raise ValueError("Only legacy_cnn supported in this quick script")

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

    print("\n--- Phase 1: Pre-training (Single Run) ---")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    ref_model_path = 'models/kfold_reference_model.pth'
    os.makedirs('models', exist_ok=True)

    for epoch in range(1, args.pretrain_epochs + 1):
        train_epoch(model, device, ref_train_loader, optimizer, criterion, epoch, phase="Pre-train")
        scheduler.step()
        
        val_loss, val_acc = evaluate(model, device, ref_val_loader, criterion, phase="Validation")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ref_model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Pre-training complete. Best model saved to {ref_model_path}")

    # 2. 5-Fold Finetuning
    print("\n--- Phase 2: 5-Fold Finetuning ---")
    finetune_dataset = RamanDataset(os.path.join(args.data_dir, 'X_finetune.npy'),
                                    os.path.join(args.data_dir, 'y_finetune.npy'),
                                    label_mapping=full_ref_dataset.label_mapping)
    test_dataset = RamanDataset(os.path.join(args.data_dir, 'X_test.npy'),
                                os.path.join(args.data_dir, 'y_test.npy'),
                                label_mapping=full_ref_dataset.label_mapping)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    ft_val_size = int(0.1 * len(finetune_dataset))
    ft_train_size = len(finetune_dataset) - ft_val_size

    test_accuracies = []

    for fold in range(1, 6):
        print(f"\n>>> Starting Fold {fold}/5 <<<")
        # Load fresh pre-trained weights
        model = get_model()
        model.load_state_dict(torch.load(ref_model_path, weights_only=True))
        
        # New random split for finetuning
        ft_train, ft_val = random_split(finetune_dataset, [ft_train_size, ft_val_size])
        ft_train_loader = DataLoader(ft_train, batch_size=args.batch_size, shuffle=True)
        ft_val_loader = DataLoader(ft_val, batch_size=args.batch_size*2, shuffle=False)
        
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        
        best_ft_val_loss = float('inf')
        best_ft_weights = None
        
        for epoch in range(1, args.finetune_epochs + 1):
            train_epoch(model, device, ft_train_loader, optimizer_ft, criterion, epoch, phase=f"Fold {fold} Finetune")
            scheduler_ft.step()
            val_loss, val_acc = evaluate(model, device, ft_val_loader, criterion, phase=f"Fold {fold} Val")
            
            if val_loss < best_ft_val_loss:
                best_ft_val_loss = val_loss
                best_ft_weights = model.state_dict()
                
        # Load best finetuned weights and evaluate on test set
        model.load_state_dict(best_ft_weights)
        print(f"\nEvaluating Fold {fold} on Test Set...")
        _, test_acc = evaluate(model, device, test_loader, criterion, phase=f"Fold {fold} Test")
        test_accuracies.append(test_acc)

    # 3. Final Results
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)
    print("\n" + "="*40)
    print("FINAL 5-FOLD CROSS VALIDATION RESULTS")
    print("="*40)
    for i, acc in enumerate(test_accuracies):
        print(f"Fold {i+1} Test Accuracy: {acc:.2f}%")
    print("-" * 40)
    print(f"Average Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
