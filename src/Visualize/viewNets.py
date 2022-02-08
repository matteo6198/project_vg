import os
from os.path import join
import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from PIL import Image
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
    x = ((x - (b + a) / 2) / (b-a)) * 2  
    return x

def apply_colormap_on_image(org_im, activation):
    """ Apply heatmap on image
    Args:
        org_img (Tensor): Original image
        activation_map (Tensor): Activation map 
    """    
    t = transforms.ToPILImage()
    activation = to_0_1(activation.squeeze(0))
    # todo rescale between -1, 1
    heatmap = colorize(activation)
    heatmap = t(heatmap)
    # Apply heatmap on image
    org_im = t(org_im.squeeze())
    heatmap = heatmap.resize(org_im.size)
    heatmap_on_image = Image.blend(org_im, heatmap, 0.6)
    return heatmap, heatmap_on_image

def get_class_activation_images(org_img, activation_map):
    out = []
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map)
    out.append((heatmap, '_Cam_Heatmap.png'))
    out.append((heatmap_on_image, '_Cam_On_Image.png'))
    return out

def save_image(im, path):
    """ Saves a torch tensor or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (torch.Tensor)):
        im = transforms.functional.to_pil_image(im.squeeze(0))
    im.save(path)

def get_img_CRN(model, img, img_transformed, net):
    with torch.no_grad():
        feat = model.backbone(img_transformed)
        N, C, W, H = feat.shape
        # build contextual reweighting mask
        if net=='CRN2':
            o1 = F.relu(model.aggregation.conv1(feat))      # 3x3 filter 
            o2 = F.relu(model.aggregation.conv2_2(F.relu(model.aggregation.conv2_1(feat))))     # equivalent to 5x5 filter
            o3 = F.relu(model.aggregation.conv3_3(F.relu(model.aggregation.conv3_2(F.relu(model.aggregation.conv3_1(feat))))))     # receptive field 7x7 filter
        else:
            m = F.avg_pool2d(feat, 2)
            o1 = F.relu(model.aggregation.conv1(m))
            o2 = F.relu(model.aggregation.conv2(m))
            o3 = F.relu(model.aggregation.conv3(m))
        m = torch.cat((o1,o2,o3), 1)
        m = model.aggregation.accumulate(m)
        if net == 'CRN':
            m = F.interpolate(m, (W, H), mode='bilinear', align_corners=False)
        x = F.normalize(feat, p=2, dim=1)
        soft_assign = model.aggregation.conv(x).view(N, model.aggregation.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign.view(N,model.aggregation.num_clusters, W, H) * m
        soft_assign = soft_assign.sum(dim=1).view((N,1,W,H))
        return get_class_activation_images(img, soft_assign.to('cpu'))

def get_img_NetVLAD(model, img, img_transformed):
    feat = model.backbone(img_transformed)
    N, C, W, H = feat.shape

    if model.aggregation.normalize_input:
        x = F.normalize(feat, p=2, dim=1)  # across descriptor dim
    # soft-assignment
    soft_assign = model.aggregation.conv(x).view(N, model.aggregation.num_clusters, -1)
    #soft_assign = F.softmax(soft_assign, dim=1)
    soft_assign = soft_assign.sum(dim=1).view((N, 1, W, H))
    return get_class_activation_images(img, soft_assign.to('cpu'))

# def get_dist(p1, p2):
#     return ((float(p1[0])-float(p2[0]))**2 + (float(p1[1])-float(p2[1]))**2) ** 0.5

def view(args, test_ds, predictions, model):
    if not(os.path.isdir(args.img_folder)):
        os.makedirs(args.img_folder)
    #### Generate recall graphs
    if build_recall_graph(args):
        print("Recall graphs generated")
    else:
        print("WARNING: impossible to build recall graphs")

    #### Extract dataset to read images
    idxs = []
    if constants.NUM_VISUALIZE >= test_ds.queries_num:
        step = 1
    else:
        # samples from equi-spatiated points
        step = test_ds.queries_num // constants.NUM_VISUALIZE
    idxs = list(range(test_ds.database_num, test_ds.database_num + test_ds.queries_num, step))

    test_ds.no_transformation = True
    query_ds = Subset(test_ds, idxs)
    query_dl = DataLoader(dataset=query_ds, num_workers=args.num_workers, batch_size=1)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(test_ds.database_utms)
    _, positives = knn.radius_neighbors(test_ds.queries_utms, 
                                     radius=args.val_positive_dist_threshold,
                                     return_distance=True,
                                     sort_results = True)
    #### build out directory
    out_dir = join(args.img_folder, test_ds.dataset_name)
    if not(os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    #### Generate images views
    id=0
    for img, idx in tqdm(query_dl, ncols=100):
        try:
            imgs = []
            imgs.append((img, '_query.png'))
            idx -= test_ds.database_num

            # get best prediction
            best_pred_idx = predictions[idx][0]
            best_pred_img, _ = test_ds.__getitem__(best_pred_idx)
            imgs.append((best_pred_img, '_best_pred.png'))
            # get best positive
            best_positive_index = positives[idx][0]
            # print(f'idx: {idx}, pos: {positives[idx][:5]}')
            best_positive_img, _ = test_ds.__getitem__(best_positive_index)
            imgs.append((best_positive_img, '_nearest_img.png'))
            # q_pos = test_ds.queries_utms[idx]
            # p_pos = test_ds.database_utms[best_pred_idx]
            # v_pos = test_ds.database_utms[best_positive_index]
            # print(f'query:{test_ds.queries_utms[idx]}, predicted pos: {test_ds.database_utms[best_pred_idx]}, nearest pos:{test_ds.database_utms[best_positive_index]}')
            # print(f'q-p: {get_dist(p_pos, q_pos)}, q-v: {get_dist(q_pos, v_pos)}, v-p: {get_dist(p_pos, v_pos)}')
            img_transformed = constants.TRANFORMATIONS['normalize'](img)
            if args.net == 'CRN' or args.net == 'CRN2':
                imgs.extend(get_img_CRN(model, img, img_transformed.to(args.device), args.net))
            elif args.net == 'NETVLAD':
                imgs.extend(get_img_NetVLAD(model, img, img_transformed.to(args.device)))

            for i, f in imgs:
                save_image(i, f'{out_dir}/{id}{f}')
        except Exception as e:
            print(f"Error on visualizing image {id}: {str(e)}")
        finally:
            id += 1
            
    test_ds.no_transformation = False
