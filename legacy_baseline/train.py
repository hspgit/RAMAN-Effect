import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import RamanDataset
from src.engine import train_epoch, evaluate
from model import LegacyCNN

def main():
    print("Running Legacy CNN from Notebook Baseline...")
    
    # 1. Load the exact same data using our RamanDataset class
    # This applies the MinMax scaling and smoothing used in our main pipeline
    # ensuring an apples-to-apples comparison of just the model architectures.
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    X_path = os.path.join(data_dir, 'X_reference.npy')
    y_path = os.path.join(data_dir, 'y_reference.npy')
    
    if not os.path.exists(X_path):
        print(f"Error: Could not find {X_path}")
        return

    print("Loading data... (This might take a moment due to preprocessing)")
    full_dataset = RamanDataset(X_path, y_path)
    
    # Notebook used 0.33 test size
    test_size = int(0.33 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], 
                                               generator=torch.Generator().manual_seed(42))
    
    # Notebook used batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    num_classes = len(np.unique(full_dataset.y))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Initialize Legacy CNN
    # The notebook used l2(0.01) on the conv layers, which we approximate with weight_decay
    model = LegacyCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Notebook used Adam with lr=2e-4
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # 3. Train
    epochs = 10 # Set to 10 for quick testing, notebook used 200
    print(f"\n--- Training Legacy CNN for {epochs} epochs ---")
    
    for epoch in range(1, epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion, epoch, phase="Train")
        if epoch % 5 == 0:
            evaluate(model, device, test_loader, criterion, phase="Val")
            
    print("\n--- Final Evaluation ---")
    evaluate(model, device, test_loader, criterion, phase="Final Test")

if __name__ == '__main__':
    main()
