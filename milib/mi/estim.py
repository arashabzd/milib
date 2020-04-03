import numpy as np
import torch
import torch.nn as nn


def logmeanexp(x, dim=None, non_diag=False):
    device = x.device
    n = x.shape[0]
    
    if non_diag:
        out = x -  torch.diag(float('inf') * torch.ones(n, device=device))
    else:
        out = x.clone()
    
    if dim is None:
        k = (n**2 - n if non_diag else n**2) * torch.ones(1, device=device)
        out = torch.logsumexp(torch.flatten(out), dim=0)
    else:
        k = (n - 1 if non_diag else n) * torch.ones(1, device=device)
        out = torch.logsumexp(out, dim=dim)
    
    return out - torch.log(k)


class TUBA(nn.Module):
    """
    Computes Tractable Unnormalized Barber Agakov (TUBA) lower bound.
    
    Params:
        non_diag: if True excludes non-diagonal elements of scores for computing marginal expectation. Default: False
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
        log_baselne: (n * 1) vector or a scaler containing an estimate of partition function for each y. Default: 0.
    """   
    
    def __init__(self, non_diag=False):
        super(TUBA, self).__init__()
        self.non_diag = non_diag

    def forward(self, scores, log_baseline=0.):
        out = scores - log_baseline
        out = torch.mean(torch.diag(out))
        out += torch.exp(logmeanexp(out, non_diag=self.non_diag))
        out += 1.
        return -out


class NWJ(nn.Module):    
    """
    Computes NWJ lower bound.
    
    Params:
        non_diag: if True excludes non-diagonal elements of scores for computing marginal expectation. Default: False
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
    """
    
    def __init__(self, non_diag=False):
        super(NWJ, self).__init__()
        self.tuba = TUBA(non_diag)

    def forward(self, scores):
        return self.tuba(scores, log_baseline=1.)


class InfoNCE(nn.Module):
    """
    Computes InfoNCE lower bound.
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
    """
    
    def __init__(self):
        super(InfoNCE, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, scores):
        device = scores.device
        n = scores.shape[0]
        targets = torch.arange(n, device=device)
        n = n * torch.ones(1, device=device)
        out = -self.loss(scores, targets) + torch.log(n)
        return -out


class JS(nn.Module):
    """
    Evaluates NWJ lower bound but backwards JS gradient.
    
    Params:
        non_diag: if True excludes non-diagonal elements of scores for computing marginal expectation. Default: False
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
    """
    
    def __init__(self, non_diag=False):
        super(JS, self).__init__()
        self.sp = nn.Softplus()
        self.nwj = NWJ(non_diag)

    def jsdiv(self, scores):
        n = scores.shape[0]
        scores_diag = torch.diag(scores)
        out = torch.mean(-self.sp(-scores_diag))
        out -= (torch.sum(self.sp(scores)) - 
                torch.sum(self.sp(scores_diag))) / (n * (n - 1))

        return out

    def forward(self, scores):
        jsd = self.jsdiv(scores)
        nwj = self.nwj(scores.clone().detach())
        out = jsd + (nwj - jsd).detach()
        return -out
    

class DV(nn.Module):
    """
    Computes DV lower bound.
    
    Params:
        non_diag: if True excludes non-diagonal elements of scores for computing marginal expectation. Default: False
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
    """
    def __init__(self, non_diag=False):
        super(DV, self).__init__()
        self.non_diag = non_diag
        
    def forward(self, scores):
        out = torch.mean(torch.diag(scores))
        out += logmeanexp(scores, non_diag=self.non_diag)
            
        return -out
            


class SMILE(nn.Module):
    """
    Computes SMILE lower bound.
    
    Params:
        tau: clamp threshold
        non_diag: if True excludes non-diagonal elements of scores for computing marginal expectation. Default: False
    
    Args:
        scores: (n * n) tensor containing f(x[i], y[j])
    """
    def __init__(self, tau, non_diag=False):
        super(SMILE, self).__init__()
        self.non_diag = non_diag
        self.tau = tau
        
    def forward(self, scores):
        t = np.exp(self.tau)
        out = torch.mean(torch.diag(scores))
        out += logmeanexp(torch.clamp(scores, -t, t), 
                               non_diag=self.non_diag)
            
        return -out
