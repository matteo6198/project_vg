
import argparse
from unittest import defaultTestLoader
from Utils import constants


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    # Other parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=3, help="num_workers for all dataloaders")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="Val/test threshold in meters")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="Train threshold in meters")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=constants.DATASETS_FOLDER, help="Path with datasets")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")

    parser.add_argument("--resume", type=str, default=False, help="reload a previuos interrupted train, it shoud contain the folder of the run like default/2022-01-02_13-18-59/")
    parser.add_argument("--net", type=str, default='default', help="specific network head to use (supported only 'GEM' or 'NETVLAD')")
    parser.add_argument("--out-dim", type=int, default=constants.ARCH_OUT_DIM['res18'], help='specify output dimensions (useful for GeM)')
    parser.add_argument("--optimizer", type=str, default='adam', help='the optimizer to use (supported "adam", "sgd")', choices=[k for k in constants.OPTIMIZERS])
    parser.add_argument("--test_only", default=False, action='store_true', help="use this with the resume argument to load the model and test it only")
    parser.add_argument("--augment", default='default', type=str, help='Augment images of train set', choices=[k for k in constants.TRANFORMATIONS])
    parser.add_argument("--netvlad_n_clusters", default=64, type=int, help="Number of clusters used with NetVlad network")
    parser.add_argument("--visual", action='store_true', default=False, help = 'set it if produce output (with test only option)')
    parser.add_argument("--whithen", action='store_true', default=False, help='Enables the whithening of the output for CRN head')

    args = parser.parse_args()
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    return args

