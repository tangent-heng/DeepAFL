import argparse
import random
import torch

from dataset import prepare_data
from federated_entity import FederatedServer, FederatedClient
from model import get_backbone_model
import numpy as np
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Framework')

    # Basic Setup
    basic_group = parser.add_argument_group('Basic Setup')
    basic_group.add_argument('--dataset', default='cifar100', type=str,
                             help='dataset name (default: cifar100)')
    basic_group.add_argument('-a', '--backbone', metavar='ARCH', default='resnet18',
                             help='backbone model architecture (default: VIT_MAE)')
    basic_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                             help='number of data loading workers (default: 4)')
    basic_group.add_argument('--gpu', default=0, type=int,
                             help='GPU id to use (default: 0)')
    basic_group.add_argument('--pretrained', dest='pretrained', action='store_true',
                             help='use pre-trained model')
    basic_group.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                     help='mini-batch size (default: 256)')


    # Data Setup
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--data_dir', default='./data', type=str, metavar='PATH',
                            help='path to dataset directory (default: ./data)')
    data_group.add_argument('--download', action='store_true', default=False,
                            help='download the dataset if not exists')
    data_group.add_argument('--seed', default=1, type=int, metavar='N',
                            help='random seed for data splitting (default: 1)')
    data_group.add_argument('--num_classes', default=100, type=int, metavar='N',
                            help='total number of classes (default: 100)')

    # Federated Learning Setup
    fed_group = parser.add_argument_group('Federated Learning Configuration')
    fed_group.add_argument('--num_clients', default=50, type=int, metavar='N',
                           help='number of clients for federated learning (default: 50)')
    fed_group.add_argument('--niid', dest='niid', action='store_true',
                           help='use non-IID data distribution among clients')
    fed_group.add_argument('--balance', dest='balance', action='store_true',
                           help='use balanced data distribution among clients')
    fed_group.add_argument('--partition', default='dir', type=str,
                           help='distribution type for non-IID setting (default: dir)')
    fed_group.add_argument('--alpha', default=0.1, type=float,
                           help='concentration parameter for Dirichlet distribution (default: 0.1)')
    fed_group.add_argument('--min_samples', default=10, type=int,
                           help='minimum number of samples per class per client in Dirichlet distribution(default: 10)')

    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model_dir', default='./weight', type=str, metavar='PATH',
                             help='path to pre-trained model weights (default: ./weight)')
    model_group.add_argument('--fe', default=512, type=int,
                             help='feature expansion size of buffer layer, 0 means no expansion (default: 0)')
    model_group.add_argument('--n_layers', default=5, type=int,
                                help='number of layers to train (default: 1)')
    model_group.add_argument('--hidden_dim', default=128, type=int,
                                help='hidden dimension of the model (default: 128)')

    # Optimization Setup
    optim_group = parser.add_argument_group('Optimization Configuration')
    optim_group.add_argument('--reg', default=0.0001, type=float,
                             help='regularization factor for analytic learning (default: 0.1)')
    optim_group.add_argument('--reg_sand', default=0.00001, type=float,
                             help='learning rate for optimization (default: 0.01)')
    optim_group.add_argument('--activation', default='tanh', type=str,
                                choices=['relu', 'gelu', 'tanh', 'sigmoid'],
                                help='activation function to use (default: tanh)')


    return parser.parse_args()


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()

    print("\n" + "="*50)
    print("EXPERIMENT CONFIGURATION")
    print("="*50)

    for arg, value in vars(args).items():
        print(f"  {arg.ljust(20)}: {value}")
    print("="*50 + "\n")

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to: {args.seed} to ensure reproducibility\n")

    # Set GPU device
    if args.gpu is not None:
        print(f"Using GPU-{args.gpu} for training")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Automatically selected device: {device}")

    print("\n" + "-"*50)
    print("Initializing model and data...")

    backbone = get_backbone_model(args)
    backbone = backbone.to(device)
    backbone.eval()
    print(f"Successfully loaded {args.backbone} backbone network")

    train_total, train_data_idx, test_data = prepare_data(args)
    print("Data preparation complete.\n")
    print("-"*50 + "\n")



    # Create server
    server = FederatedServer(
        backbone=backbone,
        train_set=train_total,
        test_set=test_data,
        args=args,
        device=device
    )

    # Create clients
    clients = []
    for idx in range(args.num_clients):
        train_dataset_idx = torch.utils.data.Subset(train_total, train_data_idx[idx])
        client = FederatedClient(
            client_id=idx,
            train_set=train_dataset_idx,
            backbone=backbone,
            args=args,
            device=device
        )
        clients.append(client)

    # start_time = time.time()

    for n in range(args.n_layers+1):
        print("\n" + "=" * 50)
        print(f"top_classifiers Training for Layer {n + 1}/{args.n_layers}")
        print("=" * 50)
        for client in clients:
            print(f"Federated Client #{client.client_id + 1}/{args.num_clients} ".ljust(40) + "-" * 10)
            corr_X, corr_Y = client.train(server.get_global_model(), n, "top_classifiers")
            server.aggregate_correlation(corr_X, corr_Y)
        print("\n" + "=" * 50)
        print(f"top_classifiers Aggregation for Layer {n + 1}/{args.n_layers}")
        print("=" * 50)
        server.update_global_model(n, "top_classifiers")

        # if n % 5 == 0:
        #     result_df.loc[n, 'train_time'] = time.time() - start_time

        if n == args.n_layers:
            print("top_classifiers training complete. Proceeding to sandwiched_layers training...\n")
            continue

        print("\n" + "=" * 50)
        print(f"sandwiched_layers Training for Layer {n + 1}/{args.n_layers}")
        print("=" * 50)
        for client in clients:
            print(f"Federated Client #{client.client_id + 1}/{args.num_clients} ".ljust(40) + "-" * 10)
            A, B= client.train(server.get_global_model(), n, "sandwiched_layers")
            server.aggregate_sandwich(A, B)
            print(f"sandwiched_layers Aggregation for Layer {n + 1}/{args.n_layers}")
        server.update_global_model(n, "sandwiched_layers")
    print("Federated learning process complete.")

    print("Evaluating model on train set...")
    server.evaluate_()

    print("Evaluating model on test set...")
    server.evaluate()





if __name__ == '__main__':
    main()