import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from Utils.constants import GEM


class GeMHeadNet(nn.Module):
    ''' module for the GeM network '''
    def __init__(self, pool, whiten):
        super(GeMHeadNet, self).__init__()
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
    
    def forward(self, x):
        # features -> pool -> norm
        x = self.norm(self.pool(x)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            x = self.norm(self.whiten(x))
        return x

class L2N(nn.Module):
    ''' Normalization with l2 norm '''
    def __init__(self, eps=GEM['eps']):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

def gem(x, p=GEM['p'], eps=GEM['eps']):
    ''' GeM pooling function '''
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

def init_gem(args):
    if GEM['whiten']:
        whiten = nn.Linear(args.features_dim, args.out_dim, bias=True)
    else:
        whiten = None
    net = GeMHeadNet(gem, whiten)
    return net