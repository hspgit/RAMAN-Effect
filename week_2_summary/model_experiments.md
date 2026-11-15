# Model Experiments & Benchmarks

During Week 2, we heavily focused on testing a wide range of model architectures to find the best fit for 1D Raman spectra (1000-point vectors). 

## Architectures Evaluated
We tested models ranging from simple Convolutional Neural Networks to advanced Attention-based architectures:

1. **Legacy CNN** (Champion - 76.50% Acc)
2. **ResNet** (69.90% Acc)
3. **Multiscale 1D-CNN** (63.73% Acc)
4. **CNN-Transformer Fusion** (60.87% Acc)
5. **1D Transformer** (54.73% Acc)

### Why the Legacy CNN won?
The shallow structure of the Legacy CNN acted as a natural "bottleneck". It was perfectly suited for the 1D spatial nature of the data, capturing local chemical peaks without having excess capacity to memorize background noise (overfit). Transformers and deep ResNets severely overfitted the training data.

## Code Snippet: FusionNet Implementation
We also implemented complex fusion networks trying to get the best of both worlds (CNN local feature extraction + Transformer global context). Here is a snippet of the FusionNet:

```python
class FusionNet(nn.Module):
    def __init__(self, seq_len=1000, num_classes=30, d_model=128):
        super().__init__()
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate sequence length after CNN
        self.seq_len_after_cnn = seq_len // 8
        
        # Transformer backend
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len_after_cnn + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # ... CNN and Transformer forward pass ...
        pass
```
