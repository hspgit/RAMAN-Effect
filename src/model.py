import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Raman1DCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(Raman1DCNN, self).__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Transformer1D(nn.Module):
    def __init__(self, num_classes=30, seq_len=1000, patch_size=100, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(Transformer1D, self).__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, 1, seq_len)
        B = x.size(0)
        
        # Patchify
        x = x.view(B, self.num_patches, self.patch_size)
        
        # Embed
        x = self.patch_embedding(x)  # (B, num_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer_encoder(x)  # (B, num_patches + 1, d_model)
        
        # Classification from CLS token
        cls_out = x[:, 0, :]  # (B, d_model)
        out = self.mlp_head(cls_out)
        
        return out

class MultiscaleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleBlock1D, self).__init__()
        # Ensure out_channels is divisible by 4
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class MultiscaleCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(MultiscaleCNN, self).__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception-style blocks
        self.block1 = MultiscaleBlock1D(16, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.block2 = MultiscaleBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.block3 = MultiscaleBlock1D(128, 256)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FusionNet(nn.Module):
    def __init__(self, num_classes=30, seq_len=1000, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.2):
        super(FusionNet, self).__init__()
        
        # Local feature extractor (CNN)
        # Keeps capacity small to avoid overfitting
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),  # 1000 -> 500
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),  # 500 -> 100
            nn.Conv1d(16, d_model, kernel_size=3, stride=1, padding=1, bias=False), # 100 -> 100
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)   # 100 -> 50
        )
        
        # After CNN, seq_len becomes 1000 / (2 * 5 * 2) = 50
        num_patches = 50
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        B = x.size(0)
        
        # 1. Local feature extraction
        x = self.cnn(x)  # (B, d_model, num_patches)
        
        # 2. Prepare for transformer
        x = x.permute(0, 2, 1)  # (B, num_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # 3. Transformer
        x = self.transformer_encoder(x)  # (B, num_patches + 1, d_model)
        
        # 4. Classification from CLS token
        cls_out = x[:, 0, :]  # (B, d_model)
        out = self.mlp_head(cls_out)
        
        return out

