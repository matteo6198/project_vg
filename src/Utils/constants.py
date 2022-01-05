DEVICE = 'cuda' # 'cuda' or 'cpu'
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/project/'  #folder where google drive project folder is mounted

TRAIN_DATASET = 'pitts30k'
TEST_DATASETS = ["pitts30k", "st_lucia"] # pitts30k or st_lucia this is used at TEST TIME

GEM = {'p':3, 'eps':1e-6, 'whiten':True}

ARCH_OUT_DIM = {
    'res18':    256
}

DATASETS_FOLDER = '/content'