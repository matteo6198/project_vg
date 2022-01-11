import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class CRNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.features_dim, 32, (3,3), padding=1)
        self.conv2 = nn.Conv2d(args.features_dim, 32, (5,5), padding=2)
        self.conv3 = nn.Conv2d(args.features_dim, 20, (7,7), padding=3)

        self.accumulate = nn.Conv2d(84, 1, (1,1))

        self.normalize = L2Norm()
        self.num_clusters = args.netvlad_n_clusters
        self.soft_ass_conv = self.conv = nn.Conv2d(args.features_dim, args.netvlad_n_clusters, kernel_size=(1, 1))
        # update output channels
        args.features_dim += self.num_clusters

    def forward(self, x):
        N, C, W, H = x.shape
        y = F.avg_pool2d(x, 3)

        o1 = self.conv1(y)
        o2 = self.conv2(y)
        o3 = self.conv3(y)
        y = torch.cat((o1,o2,o3), 1)
        y = self.accumulate(y)
        del o1,o2,o3
        y = F.interpolate(y, (W, H),mode='bilinear', align_corners=True)
        # residual
        z = self.normalize(x)
        soft_assign = self.conv(z).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1).view(N,self.num_clusters, W, H)
        out = torch.cat((z, soft_assign * y), 1)
        return out


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)