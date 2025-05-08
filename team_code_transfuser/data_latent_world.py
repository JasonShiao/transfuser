import os
import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io

class LatentWorldDataset(Dataset):
    def __init__(self, data_npy_path):
        self.data_npy_path = data_npy_path
        self.data = np.load(data_npy_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ret_data = self.data[idx]
        return ret_data
