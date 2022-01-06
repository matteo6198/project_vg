import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(path):
    if not(os.path.isdir(path)):
        raise RuntimeError("Unknown path")
    coords = {}
    paths = ['train', 'test', 'val']
    for p in paths:   
        img_path = path + p +'/database'
        if os.path.isdir(img_path):
            coords[p] = None
            for f in os.listdir(img_path):
                fields = f.split('@')
                lat, long = float(fields[6]), float(fields[5])
                if coords[p] is None:
                    coords[p] = np.array([lat,long])
                else:
                    coords[p] = np.vstack([coords[p], np.array([lat,long])])
    return coords, stats(coords)

def stats(coords):
    vett = None
    for t in coords:
        if vett is None:
            vett = coords[t]
        else:
            vett= np.vstack([vett, coords[t]])
    max_x = vett[:,0].max()
    max_y = vett[:,1].max()
    min_x = vett[:,0].min()
    min_y = vett[:,1].min()
    return (min_x, max_x, min_y, max_y)

def print_img(base_path, coords, bbox, out):
    plt.figure()
    img = plt.imread(base_path)
    fig, ax = plt.subplots(figsize=(8,7))
    labels = []
    for t in coords:
        x = coords[t][:,0]
        y = coords[t][:,1]
        ax.scatter(x,y, alpha=1, s=5)
        labels.append(t)
    ax.legend(labels)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(img, zorder=0,extent = bbox, aspect= 'equal')
    plt.savefig(out)

        

if __name__=='__main__':
    path = 'pitts30k/images/'
    coords, s = load_data(path)
    print(s)
    print_img('pitts.png',coords, s, 'pitts_out.png')

    
    path = 'st_lucia/images/'
    coords, s = load_data(path)
    print(s)
    print_img('st_lucia.png',coords, s, 'st_lucia_out.png')