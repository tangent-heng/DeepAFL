
from torchvision.transforms import transforms
import torchvision.datasets as datasets

from dataset.data_download import download_tinyimagenet
from dataset.image_folder import TinyImageNet





def tinyimagenet_dataset(args):
    """
    Get train and test datasets for TinyImageNet.
    
    Args:
        args: Command line arguments containing:
            data (str): Root directory to store the dataset
            download (bool): Whether to download the dataset if not exists
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    if args.download:
        download_tinyimagenet(args.data_dir)
    
    # Define transformations
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # Create datasets
    train_dataset = TinyImageNet(root=args.data_dir, split='train', transform=transform_train)
    test_dataset = TinyImageNet(root=args.data_dir, split='val', transform=transform_test)

    return train_dataset, test_dataset

def cifar100_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    train_dataset = datasets.CIFAR100(
        root=args.data_dir + "/cifar100", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(
        root=args.data_dir + "/cifar100", train=False, download=True, transform=train_transform)

    return train_dataset, val_dataset


def cifar10_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])


    train_dataset = datasets.CIFAR10(
        root=args.data_dir + "/cifar10", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=args.data_dir + "/cifar10", train=False, download=True, transform=train_transform)

    return train_dataset, val_dataset

