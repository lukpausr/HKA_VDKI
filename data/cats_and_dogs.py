# Datenset zum Testen der Modelle bevor das Datenset mit 
# den Zwergkaninchen erstellt ist.

import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl
import os
from torch.utils.data import Dataset
from PIL import Image

# CIFAR10 Dataset mit nur Katzen und Hunden
class BinaryCIFARDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=2, transform=None, persistent_workers=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        # Get the full binary CIFAR10 dataset for train and test
        full_train_dataset = get_binary_cifar(train=True, transform=self.transform)
        full_test_dataset = get_binary_cifar(train=False, transform=self.transform)

        # Split the training set into train and validation (e.g., 80% train, 20% val)
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Use the test set as is
        self.test_dataset = full_test_dataset

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

# Labels in 0 (cat) und 1 (dog) umwandeln
class BinaryCIFAR(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, 0 if y == 3 else 1
    def __len__(self):
        return len(self.subset)

def get_binary_cifar(train=True, transform=None):
    if transform is None:
        transform = transforms.ToTensor()
    dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)
    binary_indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
    binary_dataset = Subset(dataset, binary_indices)
    binary_dataset = BinaryCIFAR(binary_dataset)
    return binary_dataset

class KaninchenDataModule(BinaryCIFARDataModule):
    def __init__(self, batch_size=32, num_workers=2, transform=None, persistent_workers=True):
        super().__init__(batch_size, num_workers, transform, persistent_workers)

    def setup(self, stage=None):   
        self.train_dataset = BinaryFolderDataset(
            folder="D:/HKA_IMS_Drive/SS25_MSYS_KAER-AI-PoseAct/21_Test_Data/Datasets_aug/train",
            transform=self.transform
        )

        self.val_dataset = BinaryFolderDataset(
            folder="D:/HKA_IMS_Drive/SS25_MSYS_KAER-AI-PoseAct/21_Test_Data/Datasets_aug/val",
            transform=self.transform
        )

        self.test_dataset = BinaryFolderDataset(
            folder="D:/HKA_IMS_Drive/SS25_MSYS_KAER-AI-PoseAct/21_Test_Data/Datasets_aug/test",
            transform=self.transform
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
    
class BinaryFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.subset = []
        for fname in os.listdir(folder):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                label = 1 if '_ok' in fname else 0 if '_nok' in fname else None
                if label is not None:
                    self.subset.append((os.path.join(folder, fname), label))

    def __getitem__(self, idx):
        path, label = self.subset[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)