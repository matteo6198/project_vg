
import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from Networks.GeMNet import init_gem

from Networks.NetVlad import NetVLAD
from src.Utils import constants

class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        if args.net == 'GEM':
            logging.info('using GeM network')
            self.aggregation = init_gem(args)
        elif args.net == 'NETVLAD':
            logging.info('using NETVLAD network')
            self.aggregation = nn.Sequential(L2Norm(),
                                         NetVLAD(num_clusters= 64, dim=256))

        else:
            logging.info("Using default avg pool network")
            self.aggregation = nn.Sequential(L2Norm(),
                                         torch.nn.AdaptiveAvgPool2d(1),
                                         Flatten())
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_backbone(args):
    backbone = torchvision.models.resnet18(pretrained=True)
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    logging.debug("Train only conv4 of the ResNet-18 (remove conv5), freeze the previous ones")
    layers = list(backbone.children())[:-3]
    backbone = torch.nn.Sequential(*layers)
    
    feat_dim = constants.FEATURES_DIM['OTH']
    try:
        feat_dim = constants.FEATURES_DIM[args.net]
    except:
        logging.info(f"WARNING: UNKNOWN FEATURES DIM: this net {args.net} has not predefined features dimensions...using default {constants.FEATURES_DIM['OTH']}")
    args.features_dim = feat_dim

    return backbone


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:,:,0,0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
