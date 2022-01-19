try:
    from Utils.constants import DRIVE_PATH
except:
    DRIVE_PATH = '../content/drive/MyDrive/Colab Notebooks/project/'
    
from os.path import join

VAL_STRING = 'Recalls on val set'
TEST_STRING = 'Recalls on < BaseDataset'

def filter_line(l, pattern):
    return pattern in l

def build_recall_vett(l):
    recalls_str = l.split('>')[1].strip()[1:]
    recalls = recalls_str.split(',')
    recalls_v = [(recalls[i].split(':')[0].strip(), recalls[i].split(':')[1].strip()) for i in range(len(recalls))]
    return recalls_v

def getRecalls(exp_run):
    # open info file
    out = {}
    info_filename = join(DRIVE_PATH, 'runs',  exp_run, 'info.log')
    print(f'Reading recalls from file {info_filename}')
    with open(info_filename, 'r') as f:
        lines = f.readlines()
        recall_lines = filter(lambda l: filter_line(l, VAL_STRING), lines)
        recalls = [build_recall_vett(l) for l in recall_lines]
        out['val'] = recalls
        test_recalls_lines = filter(lambda l: filter_line(l, TEST_STRING), lines)
        test_recalls = [build_recall_vett(l) for l in test_recalls_lines]
        out['test'] = test_recalls
    return out

