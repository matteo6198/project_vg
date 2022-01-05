
import torch
import shutil
from os.path import join

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.output_folder, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.output_folder, "/best_model.pth"))

def resume_train(path, model, optimizer):
    ''' 
    resume the training from the last epoch reached
    returns: last_epoch_reached, best_r5, not_improved_num
    '''
    obj = torch.load(path + '/last_model.pth')
    model.load_state_dict(obj['model_state_dict'])
    optimizer.load_state_dict(obj['optimizer_state_dict'])
    epoch = obj['epoch_num'] + 1
    best_r5 = obj['best_r5']
    not_improved_num = obj['not_improved_num']
    return epoch, best_r5, not_improved_num
