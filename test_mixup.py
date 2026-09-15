import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.engine import train_epoch

def test():
    # Mock data (1D Raman spectra, 1000 points)
    X = torch.randn(32, 1, 1000)
    y = torch.randint(0, 30, (32,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Mock model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1000, 30)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print("Testing without mixup...")
    train_epoch(model, 'cpu', loader, optimizer, criterion, 1, mixup_alpha=0.0)
    
    print("Testing with mixup...")
    train_epoch(model, 'cpu', loader, optimizer, criterion, 1, mixup_alpha=0.2)
    
    print("Success!")

if __name__ == '__main__':
    test()
