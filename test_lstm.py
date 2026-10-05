import torch
import torch.nn as nn

lstm = nn.LSTM(1, 10, num_layers=2, batch_first=True, bidirectional=True)
x = torch.randn(32, 100, 1) # batch, seq_len, features
out, (h_n, c_n) = lstm(x)

print(out.shape)
print(h_n.shape)
forward_hidden = h_n[-2, :, :]
backward_hidden = h_n[-1, :, :]
print(forward_hidden.shape)

cat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
print(cat_hidden.shape)
