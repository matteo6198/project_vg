import os
import sys
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torchvision.transforms as t
from tqdm import tqdm

TRANSFORMS = {
    '1.5x': t.Resize((720, 960), interpolation=3),
    '0.5x': t.Resize((240, 320), interpolation=3)
}

DATASETS_FOLDERS =[
    '/content/pitts30k',
    '/content/st_lucia'
] 

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")

def resize_dataset(path, transforms):
    paths = glob(join(path, "**", "*.jpg"), recursive=True)
    for p in tqdm(paths):
        img = path_to_pil_img(p)
        t_img = TRANSFORMS[transforms](img)
        t_img.save(p)

if __name__ == '__main__':
    try:
        t_str = sys.argv[1]
        if not(t_str in TRANSFORMS):
            print(f'Unknown transform: possible values are ({TRANSFORMS})')
    except:
        print('Error: usage python3 resize_all_img.py <transform_name>')
        exit()
    
    for p in DATASETS_FOLDERS:
        resize_dataset(p, t_str)