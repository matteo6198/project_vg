import torch
import torchvision.transforms as transforms

DEVICE = 'cuda' # 'cuda' or 'cpu'
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/project/'  #folder where google drive project folder is mounted

TRAIN_DATASET = 'pitts30k'
TEST_DATASETS = ["pitts30k", "st_lucia"] # pitts30k or st_lucia this is used at TEST TIME

GEM = {'p':3, 'eps':1e-6, 'whiten':True}

ARCH_OUT_DIM = {
    'res18':    256
}

DATASETS_FOLDER = '/content'

MOMENTUM = 0.9
def getSGD(params, lr=1e-5):
    return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM)
    
OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd':  getSGD
}

TRANFORMATIONS = {
    'default': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'flip': transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'rotate': transforms.Compose([transforms.ToTensor(), transforms.RandomRotation([-30, 30]), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    'flip_rotate':transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation([-30, 30]), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
}

FEATURES_DIM = {
    'NETVLAD':  16384,
    'GEM':      256,
    'CRN':      16384,
    'OTH':      256
}