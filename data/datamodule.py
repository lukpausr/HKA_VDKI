# datamodule.py

# required imports
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import SmallAnimalsDataset

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
        """
        Initializes the data module with the specified data directory and batch size.

        Args:
            data_dir (str): Path to the directory containing the data.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Prepares the data for use in the data module.

        This method is typically used for tasks such as downloading, splitting, 
        or preprocessing data. However, in this case, it is not required because 
        the data is already prepared through the custom dataset. This method is 
        only called on a single GPU/TPU in distributed settings.
        """
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # not required because our data is already prepared through the custom dataset
        pass

    # Load the datasets
    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing stages.
        Args:
            stage (str, optional): The current stage of the process. Can be "fit", "test", 
                or None. Defaults to None.
                - "fit": Prepares the train and validation datasets.
                - "test": Prepares the test dataset.
                - None: Prepares all datasets (train, validation, and test).
        Attributes:
            train_dataset (SmallAnimalsDataset): Dataset instance for training.
            val_dataset (SmallAnimalsDataset): Dataset instance for validation.
            test_dataset (SmallAnimalsDataset): Dataset instance for testing.
        """
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
        """
        Creates and returns a DataLoader for the training dataset.

        This DataLoader is configured to load data from the training dataset with the specified
        batch size, number of worker processes, and shuffling enabled. Persistent workers are
        used to improve performance by keeping worker processes alive between epochs.

        Returns:
            DataLoader: A PyTorch DataLoader instance for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=True,
        )

    # Define the validation dataloader
    def val_dataloader(self):
        """
        Creates and returns a DataLoader for the validation dataset.

        The DataLoader is configured with the validation dataset, batch size, 
        and number of workers based on the available CPU cores. It is set to 
        not shuffle the data and uses persistent workers for efficient data 
        loading.

        Returns:
            DataLoader: A PyTorch DataLoader instance for the validation dataset.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )

    # Define the test dataloader
    def test_dataloader(self):
        """
        Creates and returns a DataLoader for the test dataset.

        The DataLoader is configured with the following parameters:
        - `dataset`: The test dataset to load data from.
        - `batch_size`: The number of samples per batch.
        - `num_workers`: The number of subprocesses to use for data loading, 
          set to the number of CPUs available.
        - `persistent_workers`: Whether to keep data loading workers alive 
          after the initial dataset loading.
        - `shuffle`: Whether to shuffle the data (set to False for the test dataset).

        Returns:
            DataLoader: A DataLoader instance for the test dataset.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            persistent_workers=True,
            shuffle=False,
        )