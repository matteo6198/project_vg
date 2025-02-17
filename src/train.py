import os
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

from Networks.NetVlad import init_netvlad
from Networks.CRNLayer import init_CRN
torch.backends.cudnn.benchmark = True  # Provides a speedup

from Utils import util
from Utils import parser
from Utils import commons
from Networks import base_network
from Datasets import datasets_ws as datasets
from Utils import constants
from test import test
from Visualize import viewNets

#### Initial setup: parser, logging...
args = parser.parse_arguments()
print(vars(args))
start_time = datetime.now()
#### Init args
if args.resume:
    args.output_folder = join(constants.DRIVE_PATH, "runs", args.resume)
    if not(os.path.isdir(args.output_folder)):
        raise FileNotFoundError(f"Folder {args.output_folder} does not exists")
    if (not(os.path.isfile(join(args.output_folder,'last_model.pth'))) or not(os.path.isfile(join(args.output_folder,'args.pth')))) and not(args.test_only):
        raise FileNotFoundError(f"Model file does not exists. You must restart training")
    resume = args.resume
    if not(args.test_only):
        args = torch.load(join(args.output_folder, 'args.pth'))
        args.resume = resume
        args.test_only = False
else:
    args.output_folder = join(constants.DRIVE_PATH, "runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    resume = False

commons.setup_logging(args.output_folder)
commons.make_deterministic(args.seed)

if not(os.path.isfile(join(args.output_folder, 'args.pth'))) and not(args.test_only):    
    torch.save(args, join(args.output_folder, 'args.pth'))
    logging.info("Saved args")

logging.info(f"Arguments: {vars(args)}")
logging.info(f"The outputs are being saved in {args.output_folder}")
if torch.cuda.device_count() <= 0:
    logging.info(f"WARNING RUNNING ON CPU: {multiprocessing.cpu_count()} CPUs")
    args.device = 'cpu'
else:
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Initialize model
model = base_network.GeoLocalizationNet(args)
model = model.to(args.device)
    
if not(args.test_only):    
    #### Creation of Datasets
    triplets_ds = datasets.TripletsDataset(args, args.datasets_folder, "pitts30k", "train", args.negs_num_per_query)
    val_ds = datasets.BaseDataset(args, args.datasets_folder, "pitts30k", "val")

    #### Init network params
    if args.net == 'NETVLAD' and not(resume):
        logging.debug("init netvlad weights")
        triplets_ds.is_inference = True
        init_netvlad(model, args, triplets_ds)
        triplets_ds.is_inference = False
        model = model.to(args.device)
    elif (args.net == 'CRN' or args.net == 'CRN2') and not(resume):    
        logging.debug(f"init {args.net} weights")
        triplets_ds.is_inference = True
        init_CRN(model, args, triplets_ds)
        triplets_ds.is_inference = False
        model = model.to(args.device)

    #### Setup Optimizer and Loss
    try:
        optimizer = constants.OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr)
    except:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logging.info("using default optimizer")
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    best_r5 = 0
    not_improved_num = 0
    start_epoch = 0

    #### Resume training if specified
    if resume:
        start_epoch, best_r5, not_improved_num = util.resume_train(args.output_folder, model, optimizer)
        logging.info(f"Resuming training at epoch {start_epoch}")
    else:
        logging.debug(f"Loading dataset Pitts30k from folder {args.datasets_folder}")
        logging.info(f"Train query set: {triplets_ds}")
        logging.info(f"Val set: {val_ds}")
        logging.info(f"Output dimension of the model is {constants.FEATURES_DIM[args.net]}")
        util.save_checkpoint(args, {"epoch_num": 0, "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(), "recalls": [0 for _ in args.recall_values], "best_r5": best_r5,
               "not_improved_num": not_improved_num
           }, True, filename="last_model.pth")
        logging.info("Saved empty model")

    #### Training loop
    for epoch_num in range(start_epoch, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0,1), dtype=np.float32)

        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num} / {loops_num}")

            # Compute triplets to use in the triplet loss
            triplets_ds.is_inference = True
            triplets_ds.compute_triplets(args, model)
            triplets_ds.is_inference = False

            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                     batch_size=args.train_batch_size,
                                     collate_fn=datasets.collate_fn,
                                     pin_memory=(args.device=="cuda"),
                                     drop_last=True)

            model = model.train()

            # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
            # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
            for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):

                # Compute features of all images (images contains queries, positives and negatives)
                features = model(images.to(args.device))
                loss_triplet = 0

                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T
                    loss_triplet += criterion_triplet(features[queries_indexes],
                                                      features[positives_indexes],
                                                      features[negatives_indexes])
                del features
                loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

                optimizer.zero_grad()
                loss_triplet.backward()
                optimizer.step()

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss_triplet.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                del loss_triplet

            logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                          f"current batch triplet loss = {batch_loss:.4f}, " +
                          f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        # Compute recalls on validation set
        recalls, recalls_str = test(args, val_ds, model)
        logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

        is_best = recalls[1] > best_r5

        # Save checkpoint, which contains all training parameters
        util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")

        # If recall@5 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(f"Improved: previous best R@5 = {best_r5:.3f}, current R@5 = {recalls[1]:.3f}")
            best_r5 = recalls[1]
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.3f}, current R@5 = {recalls[1]:.3f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break


    logging.info(f"Best R@5: {best_r5:.3f}")
    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.output_folder, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)


for test_dataset in constants.TEST_DATASETS:
    test_ds = datasets.BaseDataset(args, args.datasets_folder, test_dataset, "test")
    logging.info(f"Test set {test_dataset}: {test_ds}")

    recalls, recalls_str = test(args, test_ds, model)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
