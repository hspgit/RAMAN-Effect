import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import RamanDataset
from src.model import Raman1DCNN
from src.engine import train_epoch, evaluate

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
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory containing the .npy files')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='directory to save model checkpoints')
    
    # Optional arguments to allow switching models dynamically in the future
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn'],
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
    ref_dataset = RamanDataset(os.path.join(args.data_dir, 'X_reference.npy'),
                               os.path.join(args.data_dir, 'y_reference.npy'))
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=True)
    
    num_classes = len(np.unique(ref_dataset.y))
    print(f"Detected {num_classes} classes.")

    # 2. Initialize Model
    if args.model_type == 'cnn':
        model = Raman1DCNN(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Train on Reference Data
    print("\n--- Starting Reference Training ---")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, ref_loader, optimizer, criterion, epoch, phase="Pre-train")
        scheduler.step()
    
    ref_model_path = os.path.join(args.save_dir, 'raman_reference_model.pth')
    torch.save(model.state_dict(), ref_model_path)
    print(f"Reference model saved to {ref_model_path}")

    # 4. Finetune Data (if available)
    finetune_x = os.path.join(args.data_dir, 'X_finetune.npy')
    finetune_y = os.path.join(args.data_dir, 'y_finetune.npy')
    
    if os.path.exists(finetune_x) and os.path.exists(finetune_y) and args.finetune_epochs > 0:
        print("\n--- Starting Finetuning ---")
        finetune_dataset = RamanDataset(finetune_x, finetune_y, label_mapping=ref_dataset.label_mapping)
        finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Optionally reduce learning rate for finetuning
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        
        for epoch in range(1, args.finetune_epochs + 1):
            train_epoch(model, device, finetune_loader, optimizer_ft, criterion, epoch, phase="Finetune")
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
        test_dataset = RamanDataset(test_x, test_y, label_mapping=ref_dataset.label_mapping)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)
        evaluate(model, device, test_loader, criterion, phase="Test")
    else:
        print("\nTest data not found, skipping evaluation.")

if __name__ == '__main__':
    main()
