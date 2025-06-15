# dataset.py

# required imports
import os
import torch
import pandas as pd
import numpy as np
# from skimage import io, transform
from PIL import Image
from torchvision.transforms import v2

import random

# Custom Dataset for Pytorch
# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class SmallAnimalsDataset(torch.utils.data.Dataset):

    # Within the constructor, we define dataset metadata (paths, labels...) and optionally apply transformations, normalization, etc.
    def __init__(self, data_dir: str, transform=None):

        # Path to the data directory
        self.data_dir = data_dir        
        
        # Generate a list containing all file names in the given directory
        self.file_list = os.listdir(data_dir)

        # Transformations to be applied to the data
        self.transform = transform

    # Return the total number of samples of the dataset
    def __len__(self):
        """
        Returns the total number of files in the dataset.

        This method overrides the `__len__` special method to provide the length
        of the dataset, which is determined by the number of entries in the 
        `file_list` attribute.

        Returns:
            int: The number of files in the dataset.
        """
        return len(self.file_list)
    
    # https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # Called by the DataLoader to get a sample from the dataset using a given index
    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset at the specified index.
        Args:
            idx (int or torch.Tensor): The index of the data sample to retrieve. If a 
                torch.Tensor is provided, it will be converted to a Python list.
        Returns:
            dict: A dictionary containing the following keys:
                - 'image': The loaded image corresponding to the specified index.
                - 'label': The label associated with the image. Currently, this is 
                  set to 0 as a placeholder. Label loading from the file path needs 
                  to be implemented.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_path).convert("RGB")

        # Check if idx is a tensor and convert to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # TODO: Implement label loading from file path
        raise NotImplementedError("This method is not yet implemented. We need to extract the label somehow")

        sample = {'image': image, 'label': 0}

        # Apply transformations to the sample if any are defined
        if self.transform:
            sample = self.transform(sample)

        # Return the sample
        return sample
    
class BinaryImageDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for loading images from a folder and assigning binary labels based on filename suffix.
    Args:
        path_to_image_folder (str): Path to the folder containing image files.
        transform (callable, optional): Optional transform to be applied on a sample.
    Attributes:
        path_to_image_folder (str): Directory containing the images.
        filepaths (list): List of image filenames in the directory.
        transform (callable, optional): Transform to apply to each image.
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Loads an image and its binary label based on filename.
            The label is 1 if the filename ends with '_ok', 0 if it ends with '_nok'.
            Raises ValueError if the filename does not match the expected pattern.
    """
    def __init__(self, path_to_image_folder, transform=None):
        self.path_to_image_folder = path_to_image_folder
        self.filepaths = os.listdir(path_to_image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_image_folder, self.filepaths[idx])
        image = Image.open(img_path).convert("RGB")

        # Check if idx is a tensor and convert to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Label is 1 if filename ends with '_ok', 0 if ends with '_nok'
        basename = os.path.basename(img_path)
        basename = basename.lower().split('.')[0]
        print(f"Processing file: {basename}")
        if '_ok_' in basename:
            label = 1
        elif '_nok_' in basename:
            label = 0
        else:
            raise ValueError(f"Filename {basename} does not contain expected substring '_ok' or '_nok'")
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
class ReducedSizeBinaryImageDataset(BinaryImageDataset):
    """
    A custom PyTorch Dataset for loading images from a folder and assigning binary labels based on filename suffix.
    Args:
        path_to_image_folder (str): Path to the folder containing image files.
        transform (callable, optional): Optional transform to be applied on a sample.
    Attributes:
        path_to_image_folder (str): Directory containing the images.
        filepaths (list): List of image filenames in the directory.
        transform (callable, optional): Transform to apply to each image.
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Loads an image and its binary label based on filename.
            The label is 1 if the filename ends with '_ok', 0 if it ends with '_nok'.
            Raises ValueError if the filename does not match the expected pattern.
    """
    def __init__(self, path_to_image_folder, transform=None):
        self.path_to_image_folder = path_to_image_folder
        self.filepaths = os.listdir(path_to_image_folder)

        random.shuffle(self.filepaths)
        reduced_size = max(1, int(len(self.filepaths) * 0.5))
        self.filepaths = self.filepaths[:reduced_size]

        self.transform = transform

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     from config.load_configuration import load_configuration

#     config = load_configuration()

#     dataset = BinaryImageDataset(
#         path_to_image_folder=config['path_to_split_aug_pics'] + '/train/',
#         transform=None  # Add any transformations if needed
#     )

#     dataset_size = len(dataset)
#     print(f"Dataset size: {dataset_size}")

#     # Example of getting an item
#     for i in range(5):  # Print first 5 items
#         image, label = dataset[i]
#         print(f"Image {i}: {image.size}, Label: {label}")

#         import matplotlib.pyplot as plt

#         plt.imshow(image)
#         plt.title(f"Label: {label}")
#         plt.axis('off')
#         plt.show()


class MultiClassImageDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for loading images from a folder and assigning binary labels based on filename suffix.
    Args:
        path_to_image_folder (str): Path to the folder containing image files.
        transform (callable, optional): Optional transform to be applied on a sample.
    Attributes:
        path_to_image_folder (str): Directory containing the images.
        filepaths (list): List of image filenames in the directory.
        transform (callable, optional): Transform to apply to each image.
    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Loads an image and its binary label based on filename.
            The label is 1 if the filename ends with '_ok', 0 if it ends with '_nok'.
            Raises ValueError if the filename does not match the expected pattern.
    """
    def __init__(self, path_to_image_folder, name_list, transform=None):
        self.path_to_image_folder = path_to_image_folder
        self.filepaths = os.listdir(path_to_image_folder)
        self.transform = transform
        self.name_list = name_list

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_image_folder, self.filepaths[idx])
        image = Image.open(img_path).convert("RGB")

        # Apply lower() and split('.')[0] to all elements in name_list
        self.name_list = [name.lower() for name in self.name_list]

        # Check if idx is a tensor and convert to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Label is 1 if filename ends with '_ok', 0 if ends with '_nok'
        
        basename = os.path.basename(img_path)
        basename = basename.lower().split('_')[0]
        # print(f"Processing file: {basename}")

        # Assign label based on whether any element of name_list is in the basename
        label = [0.0] * len(self.name_list)  # Initialize label with zeros
        if basename in self.name_list:
            label[self.name_list.index(basename)] = 1.0
        else:
            label[0] = 1.0  # Default to first class if no match found

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
    
class MultiClassImageDataset_Bunnies(torch.utils.data.Dataset):
    """
    PyTorch Dataset for multiclass classification of bunny images.
    Expects a directory structure:
        root/
            Apollo/
            Aster/
            Helios/
            Nyx/
            other/
            Selene/
    Each subfolder contains images of that class. The folder name is the label.
    Args:
        root_dir (str): Path to the root directory (e.g., 'Train', 'Test', or 'Val').
        class_names (list): List of class names (folder names).
        transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform

        self.samples = []
        for idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = [0] * len(self.class_names)
                    label[idx] = 1.0
                    self.samples.append((os.path.join(class_folder, fname), torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.load_configuration import load_configuration
    import random
    from collections import Counter

    config = load_configuration()

    # dataset = MultiClassImageDataset_Bunnies(
    #     root_dir=config['path_to_bunnie_data'],
    #     transform=None,
    #     class_names=['Apollo', 'Aster', 'Helios', 'Nyx', 'other', 'Selene']
    # )

    # dataset_size = len(dataset)
    # print(f"Dataset size: {dataset_size}")
    # random.shuffle(dataset.samples)
    # label_counts = Counter([label for _, label in dataset.samples])
    # for class_idx, count in label_counts.items():
    #     class_name = dataset.class_names[class_idx]
    #     print(f"Class '{class_name}': {count} samples")
    # # Example of getting an item
    # for i in range(5):  # Print first 5 items
    #     image, label = dataset[i]
    #     print(f"Image {i}: {image.size}, Label: {label}")

    #     # import matplotlib.pyplot as plt

    #     # plt.imshow(image)
    #     # plt.title(f"Label: {label}")
    #     # plt.axis('off')
    #     # plt.show()


    # dataset = MultiClassImageDataset(
    #     path_to_image_folder=config['path_to_bunnie_data'],
    #     transform=None,
    #     name_list=['Apollo', 'Aster', 'Helios', 'Nyx', 'other', 'Selene']
    # )
    # print(dataset.name_list)
    # print([name.lower().split('.')[0] for name in dataset.name_list])