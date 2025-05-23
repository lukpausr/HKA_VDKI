# dataset.py

# required imports
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
# from scipy.ndimage import gaussian_filter1d

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




        # TODO: Change code to allow dynamic loading of data instead of loading all data at once
        # TODO: Change code to use images instead of csv files

        # Read all files in directory and store them in a list
        # for file in self.file_list:

            # Read data from csv file
            # temp_data = pd.read_csv(data_dir + file)
            
            # add gaussian distribution over peaks with width of 10
            # reason: the loss function can handle the peaks better when they have a larger range / area for the loss function to work with
            # for feature in Param.feature_list:
            #     if(feature == 'P-peak' or feature == 'R-peak' or feature == 'T-peak'):
                    
            #         # add gaussian distribution over peaks with width of 10 // use constant to extend data by 0s when filtering with guassian
            #         # temp_data[feature] = gaussian_filter1d(np.float64(temp_data[feature]), sigma=10, mode='constant')
            #         # normalize between 0 and 1
            #         max_val = max(temp_data[feature])
            #         if(max_val > 0):
            #             temp_data[feature] = temp_data[feature] * (1/max_val)

            #         # Print Data with matplotlib
            #         #import matplotlib.pyplot as plt
            #         #plt.plot(temp_data[feature])
            #         #plt.show()
            
            # add data to list
            # self.data.append(temp_data)

            # # Print the amount of loaded files every 1000 files for better overview during loading
            # if len(self.data) % 1000 == 0:
            #     print(f"DATASET: Loaded {len(self.data)} of {len(self.file_list)} files")

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

        image_path = os.path.join(self.data_dir, self.file_list[idx])
        image = io.imread(image_path)
        
        # TODO: Implement label loading from file path
        raise NotImplementedError("This method is not yet implemented. We need to extract the label somehow")

        sample = {'image': image, 'label': 0}

        # Apply transformations to the sample if any are defined
        if self.transform:
            sample = self.transform(sample)

        # Return the sample
        return sample