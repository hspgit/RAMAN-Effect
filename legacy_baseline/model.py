import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyCNN(nn.Module):
    """
    PyTorch implementation of the 6-layer CNN from NormalizedModelWithBootstrapping.ipynb
    """
    def __init__(self, num_classes=30):
        super(LegacyCNN, self).__init__()
        
        # Keras Conv1D(..., padding='same') with kernel_size=3 is equivalent to padding=1 in PyTorch
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.AvgPool1d(3, stride=3) # Keras AveragePooling1D(3) default stride is pool_size
        
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.AvgPool1d(3, stride=3)
        
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.AvgPool1d(3, stride=3)
        
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.AvgPool1d(3, stride=3)
        
        self.conv5 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)
        self.pool5 = nn.AvgPool1d(3, stride=3)
        
        self.conv6 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Keras GlobalAveragePooling1D
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Note: Keras layers were applied as Conv -> ReLU -> BatchNorm -> AvgPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.bn5(x)
        x = self.pool5(x)
        
        x = self.conv6(x)
        x = F.relu(x)
        x = self.bn6(x)
        x = self.global_pool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        # Softmax is omitted here because we use nn.CrossEntropyLoss in PyTorch
        return x
