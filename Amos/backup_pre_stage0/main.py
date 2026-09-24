import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import RamanDataset
from src.engine import evaluate, train_epoch
from src.model import Raman1DCNN


def main():
    parser = argparse.ArgumentParser(description='Deep Learning for Raman Spectra Classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs for reference training (default: 15)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='number of epochs for finetuning (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 penalty weight decay (default: 0.01)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory containing the .npy files')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='directory to save model checkpoints')
    
    parser.add_argument('--val-split', type=float, default=0.0,
                        help='fraction of reference data to use for validation (default: 0.0)')
    parser.add_argument('--augment', action='store_true',
                        help='enable on-the-fly data augmentation during training')
    
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='alpha parameter for mixup augmentation (default: 0.0, i.e., disabled)')
    
    # Optional arguments to allow switching models dynamically in the future
    parser.add_argument('--model-type', type=str, default='resnet', choices=['resnet', 'legacy_cnn', 'transformer', 'multiscale_cnn', 'fusion'],
                        help='type of model to use')
                        
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 1. Load Reference Data
    print("Loading reference dataset...")
    full_ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                                    os.path.join(args.data_dir, 'y_reference.npy'))
    
    num_classes = len(np.unique(full_ref_dataset.y))
    print(f"Detected {num_classes} classes.")
    
    from src.dataset import AugmentedDataset
    
    if args.val_split > 0:
        from torch.utils.data import random_split
        val_size = int(args.val_split * len(full_ref_dataset))
        train_size = len(full_ref_dataset) - val_size
        ref_dataset, val_dataset = random_split(full_ref_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        
        # Apply augmentation only to training data
        train_dataset_wrapped = AugmentedDataset(ref_dataset, augment=args.augment)
        val_dataset_wrapped = AugmentedDataset(val_dataset, augment=False)
        
        ref_loader = DataLoader(train_dataset_wrapped, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_wrapped, batch_size=args.batch_size * 2, shuffle=False)
        print(f"Split reference data: {train_size} train / {val_size} val")
    else:
        train_dataset_wrapped = AugmentedDataset(full_ref_dataset, augment=args.augment)
        ref_loader = DataLoader(train_dataset_wrapped, batch_size=args.batch_size, shuffle=True)
        val_loader = None

    # 2. Initialize Model
    if args.model_type == 'resnet':
        model = Raman1DCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'legacy_cnn':
        from legacy_baseline.model import LegacyCNN
        model = LegacyCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'transformer':
        from src.model import Transformer1D
        model = Transformer1D(num_classes=num_classes).to(device)
    elif args.model_type == 'multiscale_cnn':
        from src.model import MultiscaleCNN
        model = MultiscaleCNN(num_classes=num_classes).to(device)
    elif args.model_type == 'fusion':
        from src.model import FusionNet
        model = FusionNet(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 3. Train on Reference Data
    print("\n--- Starting Reference Training ---")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    ref_model_path = os.path.join(args.save_dir, 'raman_reference_model.pth')
    
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, ref_loader, optimizer, criterion, epoch, phase="Pre-train", mixup_alpha=args.mixup_alpha)
        scheduler.step()
        
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, device, val_loader, criterion, phase="Validation")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ref_model_path)
                print(f"--> Best validation loss improved. Saved model to {ref_model_path}")
            else:
                patience_counter += 1
                print(f"--> No improvement in validation loss. Patience: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}! Reverting to best model.")
                break
    
    if val_loader is None:
        torch.save(model.state_dict(), ref_model_path)
        print(f"Reference model saved to {ref_model_path}")
    elif os.path.exists(ref_model_path):
        model.load_state_dict(torch.load(ref_model_path, weights_only=True))
        print("Loaded best reference model for finetuning.")

    # 4. Finetune Data (if available)
    finetune_x = os.path.join(args.data_dir, 'X_finetune.npy')
    finetune_y = os.path.join(args.data_dir, 'y_finetune.npy')
    
    if os.path.exists(finetune_x) and os.path.exists(finetune_y) and args.finetune_epochs > 0:
        print("\n--- Starting Finetuning ---")
        finetune_dataset = RamanDataset(finetune_x, finetune_y, label_mapping=full_ref_dataset.label_mapping)
        finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Optionally reduce learning rate for finetuning
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        
        for epoch in range(1, args.finetune_epochs + 1):
            train_epoch(model, device, finetune_loader, optimizer_ft, criterion, epoch, phase="Finetune", mixup_alpha=args.mixup_alpha)
            scheduler_ft.step()
            
        ft_model_path = os.path.join(args.save_dir, 'raman_finetuned_model.pth')
        torch.save(model.state_dict(), ft_model_path)
        print(f"Finetuned model saved to {ft_model_path}")
    else:
        print("\nSkipping finetuning (files not found or finetune-epochs=0).")

    # 5. Evaluate on Test Data
    test_x = os.path.join(args.data_dir, 'X_test.npy')
    test_y = os.path.join(args.data_dir, 'y_test.npy')
    
    if os.path.exists(test_x) and os.path.exists(test_y):
        print("\n--- Evaluating on Test Set ---")
        test_dataset = RamanDataset(test_x, test_y, label_mapping=full_ref_dataset.label_mapping)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
        evaluate(model, device, test_loader, criterion, phase="Test")
    else:
        print("\nTest data not found, skipping evaluation.")

if __name__ == '__main__':
    main()
