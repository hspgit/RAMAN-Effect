# RNN Implementation

This document details the exact PyTorch implementation of the `RamanLSTM` model added to `src/model.py`.

## The `RamanLSTM` Class

```python
import torch
import torch.nn as nn

class RamanLSTM(nn.Module):
    """
    A sequential model (RNN/LSTM) for Raman spectroscopy data.
    Treats the spectrum as a sequence of intensities.
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=30, dropout=0.3):
        super(RamanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer (batch_first=True means input should be of shape [batch, seq_len, features])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0)
        
        # We multiply hidden_size by 2 because it's a bidirectional LSTM
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Your data comes in as [batch, channels(1), seq_len] (e.g., [32, 1, 1000])
        # LSTM expects [batch, seq_len, features(1)]
        x = x.permute(0, 2, 1)
        
        # out shape: (batch, seq_len, hidden_size * 2)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # Extract the hidden state from the last layer for both directions
        # h_n[-2, :, :] is the forward direction's last layer
        # h_n[-1, :, :] is the backward direction's last layer
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        
        # Concatenate forward and backward final states
        out = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        out = self.dropout(out)
        out = self.fc(out)
        return out
```

## Integration Details
The model was fully integrated into the project pipeline:
- Added `--model-type rnn` argument to `main.py` and `evaluate_kfold.py`.
- Instantiated via standard PyTorch training loops, validating that tensor dimensions `[batch, 1, 1000]` map correctly to the 30-class output logits.
