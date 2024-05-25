# import some packages you need here
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here
        with open(input_file, 'r') as file:
            self.text = file.read()
            
        # construct character dictionary {index:character}
        self.chars = sorted(set(self.text))
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
        
        # character indices
        self.text_indices = [self.char2idx[char] for char in self.text]
        
        # sequence length
        self.seq_length = 30
        
        self.data = []
        self.targets = []
        for i in range(0, len(self.text_indices) - self.seq_length):
            self.data.append(self.text_indices[i:i + self.seq_length])
            self.targets.append(self.text_indices[i + 1:i + self.seq_length + 1])    

            
    def __len__(self):

        # write your codes here
        return len(self.data)


    def __getitem__(self, idx):

        # write your codes here
        input_sq = torch.tensor(self.data[idx], dtype=torch.long)
        target_sq = torch.tensor(self.targets[idx], dtype=torch.long)

        return input_sq, target_sq

if __name__ == '__main__':

    # write test codes to verify your implementations
    dataset = Shakespeare('C:\\Users\\TEEM\\Desktop\\shakespears\\shakespeare_train.txt') 
    print(f"Dataset size: {len(dataset)}")

    

    

    