import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


class BinaryImageDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert("RGB")
        # Label is 1 if filename ends with '_ok', 0 if ends with '_nok'
        basename = os.path.basename(img_path)
        basename = basename.lower().split('.')[0]
        if basename.endswith('_ok'):
            label = 1
        elif basename.endswith('_nok'):
            label = 0
        else:
            raise ValueError(f"Filename {basename} does not match expected pattern '_ok' or '_nok'")
        
        if self.transform:
            image = self.transform(image)
        return image, label


class BinaryImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=2, transform=None, persistent_workers=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        test_files = glob(os.path.join(self.data_dir, "test/", "*.*"))
        train_files = glob(os.path.join(self.data_dir, "train/", "*.*"))
        val_files = glob(os.path.join(self.data_dir, "val/", "*.*"))

        self.train_dataset = BinaryImageDataset(train_files, self.transform)
        self.val_dataset = BinaryImageDataset(val_files, self.transform)
        self.test_dataset = BinaryImageDataset(test_files, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
