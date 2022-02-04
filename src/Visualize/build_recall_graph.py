import os
import argparse
from os.path import join
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from log_utils import getRecalls
else:
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
    if not(os.path.isfile(join(args.output_folder, 'info.log'))):
        return False
    #### Read recalls from info.log
    recalls_dict = getRecalls(join(*args.output_folder.split('/')[7:]))
    if len(recalls_dict) <= 0:
        return False
    #### Build output directory
    out_dir = join(args.img_folder, 'recalls')
    if not(os.path.isdir(out_dir)):
        os.makedirs(out_dir)
    #### Generate graphs
    try:
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
    except:
        return False
    return True

def get_best_recall(recalls):        
    r5 = [float(recs[1][1]) for recs in recalls]
    max_r5 = max(r5)
    best_recalls = list(filter(lambda s: float(s[1][1])==max_r5, recalls))[0]
    return best_recalls

def make_comapre_recall_graph(runs, legend, out_folder):
    all_recalls = {'val':[], 'test_pitts30k':[], 'test_st_lucia':[]}
    try:
        for r in runs:
            recalls = getRecalls(r)
            try:
                all_recalls['val'].append(get_best_recall(recalls['val']))
            except:
                print("Can't build validation graph")
                #del all_recalls['val']

            all_recalls['test_pitts30k'].append(recalls['test'][0])
            all_recalls['test_st_lucia'].append(recalls['test'][1])
    except Exception as e:
        print(f'ERROR: Unable to extract recalls: {e}')
        exit()
    #### build graphs
    for k in all_recalls:
        recalls = all_recalls[k]
        plt.figure()
        for i, r in enumerate(recalls):
            x, y = extract_recalls(r)
            plt.plot(x, y, '*-', label=legend[i])
            plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.xlabel('Recalls')
        plt.savefig(join(out_folder, str(k) + '_recalls_graph.png')) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot recall graph for the specified runs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--exp_runs", type=str, nargs='+', required=True, default=[], help="Runs of the networks to plot in the form <exp_name>/<date_time> (ex crn/lr4/2022-01-14_09-44-49)")
    parser.add_argument('--legend', type=str, nargs='+', required=True, default=[], help='The legend names to be displayed')
    parser.add_argument('--out_dir', type=str, default='.', help='The output directory. If not specified the current directory is used')

    args = parser.parse_args()
    if len(args.exp_runs) != len(args.legend):
        raise ValueError(f'The number of legend voices {len(args.legend)} is different from the number of runs {len(args.exp_runs)}')
    if not(os.path.isdir(args.out_dir)):
        print(f"Creating directory {args.out_dir}...")
        os.makedirs(args.out_dir)
    make_comapre_recall_graph(args.exp_runs, args.legend, args.out_dir)