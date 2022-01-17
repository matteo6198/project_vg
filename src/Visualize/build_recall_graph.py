import os
from os.path import join
import matplotlib.pyplot as plt

from Visualize.log_utils import getRecalls

def make_graph(x, y, out_filename):
    plt.figure()
    plt.grid(True)
    plt.plot(x, y, '+-')
    plt.xticks(x)
    plt.savefig(out_filename)

def extract_recalls(recalls):    
    # extract values
    x, y = [], []
    [(x.append(int(v[0].split('@')[1])), y.append(float(v[1]))) for v in recalls]
    return x,y

def build_recall_graph(args):
    out_dir = join(args.output_folder, 'img', 'recalls')
    if not(os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    recalls_dict = getRecalls(args)

    # val recalls
    val_recalls = recalls_dict['val']
    r5 = [float(recs[1][1]) for recs in val_recalls]
    make_graph(range(1, len(r5)+1), r5, join(out_dir, 'r5_train_graph.png'))
    # best val recall string
    max_r5 = max(r5)
    best_val_recalls = list(filter(lambda s: float(s[1][1])==max_r5, val_recalls))[0]
    x, y = extract_recalls(best_val_recalls)
    make_graph(x, y, join(out_dir, 'best_val_recalls_graph.png'))

    # test
    plt.figure()
    plt.grid(True)
    x, y = extract_recalls(recalls_dict['test'][0])
    plt.plot(x, y, '+-', label='pitts30k')
    x, y = extract_recalls(recalls_dict['test'][1])
    plt.plot(x, y, '*-', label='st lucia')
    plt.legend()
    plt.xticks(x)
    plt.savefig(join(out_dir, 'test_recalls.png'))

