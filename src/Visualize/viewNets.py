from os.path import join
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image
import numpy as np
import math

from Networks import base_network
from Datasets import datasets_ws as datasets
from Utils import constants
from Visualize.build_recall_graph import build_recall_graph

def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    ''' Converts a one-channel grayscale image to a color heatmap image '''
    if x.dim() == 2:
        torch.unsqueeze(x, 0, out=x)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        raise ValueError("Shape too big (max 3 dim)")
    return cl
def to_0_1(x):
    a, b = x.min(), x.max()
    x = x / (b-a) - (b + a) / 2 
    return x

def apply_colormap_on_image(org_im, activation):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map 
        colormap_name (str): Name of the colormap
    """    
    t = transforms.ToPILImage()
    activation = to_0_1(-activation.squeeze(0))
    # todo rescale between -1, 1
    heatmap = colorize(activation)
    heatmap = t(heatmap)
    # Apply heatmap on image
    org_im = t(org_im.squeeze())
    heatmap = heatmap.resize(org_im.size)
    heatmap_on_image = Image.blend(org_im, heatmap, 0.6)
    return heatmap, heatmap_on_image

def get_class_activation_images(org_img, activation_map):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    out = []
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map)
    out.append((heatmap, '_Cam_Heatmap.png'))
    out.append((heatmap_on_image, '_Cam_On_Image.png'))
    return out

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (torch.Tensor)):
        im = transforms.functional.to_pil_image(im.squeeze(0))
    im.save(path)

def get_img_CRN(model, img, img_transformed):
    with torch.no_grad():
        feat = model.backbone(img_transformed)
        N, C, W, H = feat.shape
        # build contextual reweighting mask
        m = F.avg_pool2d(feat, 2)
        o1 = F.relu(model.aggregation.conv1(m))
        o2 = F.relu(model.aggregation.conv2(m))
        o3 = F.relu(model.aggregation.conv3(m))
        m = torch.cat((o1,o2,o3), 1)
        m = model.aggregation.accumulate(m)
        m = F.interpolate(m, (W, H), mode='bilinear', align_corners=False)

        return get_class_activation_images(img, m.to('cpu'))

def view(args, test_ds, predictions, model):
    if not(os.path.isdir(args.img_folder)):
        os.makedirs(args.img_folder)
    build_recall_graph(args)

    num_queries_to_get = min(constants.NUM_VISUALIZE, test_ds.queries_num)

    test_ds.no_transformation = True
    query_ds = Subset(test_ds, range(eval_ds.database_num, eval_ds.database_num+num_queries_to_get))
    query_dl = DataLoader(dataset=query_ds, num_workers=args.num_workers, batch_size=1)
    predictions = predictions[:num_queries_to_get]
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(test_ds.database_utms)
    _, positives = knn.radius_neighbors(test_ds.queries_utms, 
                                     radius=args.val_positive_dist_threshold,
                                     return_distance=True,
                                     sort_results = True)
    # build out directory
    out_dir = join(args.img_folder, test_dataset)
    if not(os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    for img, idx in tqdm(query_dl, ncols=100):
        imgs = []
        imgs.append((img, '_query.png'))

        # get best prediction
        best_pred_idx = predictions[idx][0]
        best_pred_img, _ = test_ds.__getitem__(best_pred_idx)
        imgs.append((best_pred_img, '_best_pred.png'))
        # get best positive
        best_positive_index = positives[idx][0]
        best_positive_img, _ = test_ds.__getitem__(best_positive_index)
        imgs.append((best_positive_img, '_nearest_img.png'))
        #save_image(img, join(out_dir, str(idx.item())+'.png'))
        img_transformed = constants.TRANFORMATIONS['normalize'](img)
        if args.net == 'CRN':
            imgs.extend(get_img_CRN(model, img, img_transformed))
        
        for i, f in imgs:
            save_image(i, f'{out_dir}/{idx.item()}{f}')
            
    test_ds.no_transformation = False
