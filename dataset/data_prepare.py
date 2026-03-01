import torch.utils.data
import numpy as np

from dataset.generate_data import *


def prepare_data(args):

    if args.dataset == "tinyimagenet":
        trainset, testset = tinyimagenet_dataset(args)
    elif args.dataset == "cifar100":
        trainset, testset = cifar100_dataset(args)
    elif args.dataset == "cifar10":
        trainset, testset = cifar10_dataset(args)
    else:
        trainset, testset = None, None
        print("Unavailable dataset!")
        return trainset, testset


    np.random.seed(args.seed)
    data_idx_train, y, statistic = separate_data(trainset, testset, args.num_clients, args.num_classes,
                                    args.niid, args.balance, args.partition, args.alpha,least_samples = args.min_samples)



    num_classes = args.num_classes
    print(num_classes)
    trainset.targets = torch.eye(num_classes)[trainset.targets]
    testset.targets = torch.eye(num_classes)[testset.targets]

    return trainset, data_idx_train, testset




def separate_data(data, data_test, num_clients, num_classes, niid=False, balance=False, partition=None, alpha = 0.1,least_samples=1,class_per_client = 10):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_label_train = data.targets
    dataset_label = np.array(dataset_label_train)

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
        #     if try_cnt > 1:
        #         print(
        #             f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                min_clients = [i for i, idx_j in enumerate(idx_batch) if len(idx_j) == min_size]
                max_clients = [i for i, idx_j in enumerate(idx_batch) if len(idx_j) == max([len(idx) for idx in idx_batch])]
                if min_size < least_samples:
                    for m in min_clients:
                        idx_batch[m] += idx_batch[max_clients[0]][:least_samples - min_size]
                        idx_batch[max_clients[0]] = idx_batch[max_clients[0]][least_samples - min_size:]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                try_cnt += 1



        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    data_idx = []
    for client in range(num_clients):
        idxs = dataidx_map[client]
        idxs = np.array(idxs)
        y[client] = dataset_label[idxs]
        data_idx.append(torch.from_numpy(idxs))
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data, data_test
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(y[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return data_idx, y, statistic





