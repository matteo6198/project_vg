from os.path import join
import os
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np

from Networks import base_network
from Datasets import datasets_ws as datasets
from Utils import constants

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation.squeeze(0).permute(2, 1, 0).numpy())
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    
    t = transforms.ToPILImage()

    heatmap = torch.from_numpy(heatmap).squeeze().permute(2, 1, 0)
    heatmap = t(heatmap)
    no_trans_heatmap = t(torch.from_numpy(no_trans_heatmap).squeeze().permute(2, 1, 0))
    # Apply heatmap on iamge
    org_im = t(org_im.squeeze())
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

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
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
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
        t = transforms.ToPILImage()
        im = t(im.squeeze(0))
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
        m = F.interpolate(F.normalize(F.interpolate(m, (W, H), mode='bilinear', align_corners=False), p=2, dim=1), (img.shape[2], img.shape[3]), mode='bilinear', align_corners=True)

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
            if args.net == 'CRN':
                imgs = get_img_CRN(model, img, img_transformed)
                for i, f in imgs:
                    save_image(i, f'{out_dir}/{idx.item()}{f}')



 
