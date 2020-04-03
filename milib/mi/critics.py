import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=0, 
                 activation=None, bn=False):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            *[x for i in range(num_layers - 1) 
              for x in [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim, bias=not bn)] + 
                       ([nn.BatchNorm1d(hidden_dim)] if bn else []) +  
                       [activation()]] +
             [nn.Linear(input_dim if num_layers == 1 else hidden_dim, output_dim, bias=not bn)] +
             ([nn.BatchNorm1d(output_dim)] if bn else [])
        )
            
        
    
    def forward(self, x):
        return self.mlp(x)


class SeparableCritic(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, hidden_dim, 
                 embed_dim, num_layers, activation, bn):
        super(SeparableCritic, self).__init__()

        self.f = MLP(input_dim_x, embed_dim, num_layers, hidden_dim, activation, bn)
        self.g = MLP(input_dim_y, embed_dim, num_layers, hidden_dim, activation, bn)

    def forward(self, x, y):
        return torch.matmul(self.f(x), self.g(y).t())


class BiLinearCritic(nn.Module):
    def __init__(self, input_dim_x, input_dim_y):
        super(BiLinearCritic, self).__init__()
        self.f = nn.Linear(input_dim_x, input_dim_y, bias=False)
        
    def forward(self, x, y):
        return torch.matmul(self.f(x), y.t())