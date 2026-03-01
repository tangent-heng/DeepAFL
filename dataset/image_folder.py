import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
from torch.utils.data import Dataset

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

    

class TinyImageNet(torch.utils.data.Dataset):
    """
    TinyImageNet dataset.
    
    Args:
        root (str): Root directory of the dataset
        split (str): 'train' or 'val'
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version
    """
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        
        self.images = []
        self.targets = []
        self.classes = []
        
        # Load data based on split
        if split == 'train':
            # Process training data
            train_dir = os.path.join(self.root, 'tiny-imagenet-200', 'train')
            self._load_train_data(train_dir)
        elif split == 'val':
            # Process validation data
            val_dir = os.path.join(self.root, 'tiny-imagenet-200', 'val')
            self._load_val_data(val_dir)
        else:
            raise ValueError(f"Split {split} not supported. Use 'train' or 'val'.")
    
    def _load_train_data(self, train_dir):
        """Load training data"""
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        self.classes = sorted(class_dirs)
        class_to_idx = {class_dir: i for i, class_dir in enumerate(self.classes)}
        
        print(f"Loading training data from {len(class_to_idx)} classes...")
        for class_dir, idx in class_to_idx.items():
            img_dir = os.path.join(train_dir, class_dir, 'images')
            img_files = [img for img in os.listdir(img_dir) if img.endswith('.JPEG')]
            for img_name in img_files:
                img_path = os.path.join(img_dir, img_name)
                self.images.append(img_path)
                self.targets.append(idx)
    
    def _load_val_data(self, val_dir):
        """Load validation data"""
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        img_to_class = {}
        
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_to_class[parts[0]] = parts[1]
        
        train_dir = os.path.join(self.root, 'tiny-imagenet-200', 'train')
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        self.classes = sorted(class_dirs)
        class_to_idx = {class_dir: i for i, class_dir in enumerate(self.classes)}
        
        img_dir = os.path.join(val_dir, 'images')
        img_files = [img for img in os.listdir(img_dir) if img in img_to_class]
        print(f"Loading validation data: {len(img_files)} images...")
        for img_name in img_files:
            img_path = os.path.join(img_dir, img_name)
            class_id = img_to_class[img_name]
            idx = class_to_idx[class_id]
            self.images.append(img_path)
            self.targets.append(idx)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path, target = self.images[index], self.targets[index]

        # Load image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Apply transforms if any
        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def __len__(self):
        return len(self.images)


