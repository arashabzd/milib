import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder
        
    def forward(self, x):
        n, l, _, c, h, w = x.shape
        out = x.reshape(n*l*l, c, h, w)
        out = self.encoder(out)
        out = out.reshape(n, l, l, -1)
        return out
        
        
class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNContext, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=True,
                          batch_first=True)
    
    def forward(self, x):
        n, l, _, _ = x.shape
        out = x.reshape(n, l*l, -1)
        out = self.gru(out)[0]
        out = out.reshape(n, l, l, 2, -1)
        return out[:, :, :, 0, :], out[:, :, :, 1, :]

    
class CPC(nn.Module):
    def __init__(self, encoder, context, critics1, critics2):
        super(CPC, self).__init__()
        
        self.encoder = encoder
        self.context = context
        self.critics1 = nn.ModuleDict(critics1)
        self.critics2 = nn.ModuleDict(critics2)
    
    def forward(self, x):
        z = self.encoder(x)
        c1, c2 = self.context(z)
        
        scores1 = []
        for p, critic in self.critics1.items():
            p = int(p)
            c_p = c1[:, :-p, :, :].clone().flatten(end_dim=-2)
            z_p = z[:, p:, :, :].clone().flatten(end_dim=-2)
            scores1.append(self.critics1[str(p)](c_p, z_p))
            
        scores2 = []
        for p, critic in self.critics2.items():
            p = int(p)
            c_p = c2[:, :-p, :, :].clone().flatten(end_dim=-2)
            z_p = z[:, p:, :, :].flip(1, 2).clone().flatten(end_dim=-2)
            scores2.append(self.critics2[str(p)](c_p, z_p))
        
        return scores1 + scores2
        
        
        
