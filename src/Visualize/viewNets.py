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
from tqdm import tqdm
from PIL import Image
import numpy as np
import math

from Networks import base_network
from Datasets import datasets_ws as datasets
from Utils import constants

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

def view(args):
    if not(os.path.isdir(args.img_folder)):
        os.makedirs(args.img_folder)

    # load best model weights
    model = base_network.GeoLocalizationNet(args)
    best_state_dict = torch.load(join(args.output_folder, 'best_model.pth'))['model_state_dict']
    model.load_state_dict(best_state_dict)
    # visualize for each dataset
    for test_dataset in constants.TEST_DATASETS:
        test_ds = datasets.BaseDataset(args, args.datasets_folder, test_dataset, "test")
        test_ds.no_transformation = True
        visual_ds = Subset(test_ds, range(constants.NUM_VISUALIZE))
        visual_dl = DataLoader(dataset=visual_ds, num_workers=args.num_workers, batch_size=1)
        # build out directory
        out_dir = join(args.img_folder, test_dataset)
        if not(os.path.isdir(out_dir)):
            os.makedirs(out_dir)

        for img, idx in tqdm(visual_dl, ncols=100):
            save_image(img, join(out_dir, str(idx.item())+'.png'))
            img_transformed = constants.TRANFORMATIONS['normalize'](img)
            imgs = []
            if args.net == 'CRN':
                imgs = get_img_CRN(model, img, img_transformed)
            else:
                raise ValueError(f'Unknown net {args.net}')
            
            for i, f in imgs:
                save_image(i, f'{out_dir}/{idx.item()}{f}')



 
