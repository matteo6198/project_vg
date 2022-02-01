import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from Networks import NetVlad
from Utils.constants import FEATURES_DIM

class CRNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.features_dim, 32, (3,3), padding=1)
        self.conv2 = nn.Conv2d(args.features_dim, 32, (5,5), padding=2)
        self.conv3 = nn.Conv2d(args.features_dim, 20, (7,7), padding=3)

        self.accumulate = nn.Conv2d(84, 1, (1,1))

        self.num_clusters = args.netvlad_n_clusters
        self.conv = nn.Conv2d(args.features_dim, args.netvlad_n_clusters, kernel_size=(1, 1))
        self.conv.bias = None
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, args.features_dim))
        # update output channels
        args.features_dim += self.num_clusters

    def forward(self, x):
        N, C, W, H = x.shape
        # build contextual reweighting mask
        m = F.avg_pool2d(x, 2)

        o1 = F.relu(self.conv1(m))
        o2 = F.relu(self.conv2(m))
        o3 = F.relu(self.conv3(m))
        m = torch.cat((o1,o2,o3), 1)
        m = self.accumulate(m)
        m = F.interpolate(m, (W, H), mode='bilinear', align_corners=False)
        # residual
        x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        shape = soft_assign.shape
        soft_assign = (soft_assign.view(N,self.num_clusters, W, H) * m).view(shape)     # element-wise multiplication of soft assign and mask
        # NetVLAD core
        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        out = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            out[:,C:C+1,:] = residual.sum(dim=-1)

        out = F.normalize(out, p=2, dim=2)  # intra-normalization
        out = out.view(x.size(0), -1)       # flatten
        out = F.normalize(out, p=2, dim=1)  # L2 normalize
        return out

    def init_params(self, centroids, descriptors):
        clstsAssign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending
        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

def init_CRN(model, args, dataset):    
    centroids, descriptors = NetVlad.get_clusters(args, dataset, model)
    model.aggregation.init_params(centroids, descriptors)