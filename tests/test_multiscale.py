import os
import sys
import torch
sys.path.append(os.getcwd())
from src.model import MultiscaleCNN

def test_multiscale_cnn():
    model = MultiscaleCNN(num_classes=30)
    x = torch.randn(2, 1, 1000)
    y = model(x)
    print("MultiscaleCNN Output shape:", y.shape)
    assert y.shape == (2, 30), f"Expected shape (2, 30) but got {y.shape}"
