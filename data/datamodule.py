# datamodule.py

# required imports
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import SmallAnimalsDataset

# Custom Data Module for Pytorch Lightning
# Source: https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html#data
# This data module automatically handles the training, test and validation data and we don't have to worry
class Animal_DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for handling Animal image data. This class is responsible for
    preparing the data, setting up datasets for different stages (train, validation, test),
    and providing data loaders for each stage.
    Attributes:
        data_dir (str): The directory where the data is stored.
        batch_size (int): The batch size to be used for data loading.
    Methods:
        prepare_data():
            Prepares the data (e.g., downloading, splitting). This method is not required
            in this case as the data is already prepared through the custom dataset.
        setup(stage=None):
            Sets up the datasets for the specified stage ('fit', 'test', or None). Assigns
            train, validation, and test datasets for use in data loaders.
        train_dataloader():
            Returns the DataLoader for the training dataset.
        val_dataloader():
            Returns the DataLoader for the validation dataset.
        test_dataloader():
            Returns the DataLoader for the test dataset.
    """
    
    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # not required because our data is already prepared through the custom dataset
        pass

    # Load the datasets
    def setup(self, stage=None):

        # the datasets instances are generated here, depending on the current stage (val/train/test split)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = SmallAnimalsDataset(data_dir=self.data_dir+'\\pd_dataset_train\\')
            self.val_dataset = SmallAnimalsDataset(data_dir=self.data_dir+'\\pd_dataset_val\\')
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = SmallAnimalsDataset(data_dir=self.data_dir+'\\pd_dataset_test\\')
            pass

    # Define the train dataloader
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=True,
        )

    # Define the validation dataloader
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

    # Define the test dataloader
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )