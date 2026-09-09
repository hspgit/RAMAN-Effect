import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd())
from src.model import Raman1DCNN

try:
    X = np.load('data/X_reference.npy')
    print("X shape:", X.shape)
    L = X.shape[1]
except Exception as e:
    print("Could not load data:", e)
    L = 2000

model = Raman1DCNN(num_classes=30)
x = torch.randn(2, 1, L)
try:
    y = model(x)
    print("Success. Output shape:", y.shape)
except Exception as e:
    print("Failed to run model:")
    print(e)
