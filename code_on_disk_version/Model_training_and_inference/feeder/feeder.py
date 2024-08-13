# sys
import os
import sys
import numpy as np
import random
import pickle

from scipy.spatial.transform import Rotation as R

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time
import yaml
# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 config_file_path,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True,
                 training=True):
        with open(config_file_path, 'r') as f:
            configs = yaml.safe_load(f)
        self.max_min_norm = configs['max_min_norm']
        self.rotation = configs['rotation']
        self.rot_min, self.rot_max = configs['rot_min'], configs['rot_max']

        self.debug = debug
        self.training = training
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        if self.training:
            if self.rotation:
                rot_list = list(range(self.rot_min, self.rot_max + 1, 45))
                rot_deg = np.random.choice(rot_list)
                r = R.from_euler('z', rot_deg, degrees=True)
                original_shape = data_numpy.shape
                data_numpy = data_numpy.reshape(3, -1).T
                data_numpy = r.apply(data_numpy)
                data_numpy = data_numpy.T.reshape(original_shape)
            
            if self.max_min_norm:
                data_numpy = (data_numpy - data_numpy.min(-2, keepdims=True)) / \
                        (data_numpy.max(-2, keepdims = True) - data_numpy.min(-2, keepdims=True) + 1e-4)

            # processing
            if self.random_choose: # all 25 window size, this func useless
                data_numpy = tools.random_choose(data_numpy, self.window_size)
            elif self.window_size > 0:
                data_numpy = tools.auto_pading(data_numpy, self.window_size)
            if self.random_move: # random rotation (-10 ~ 10) + random scale (0.9 ~ 1.1). all small.
                data_numpy = tools.random_move(data_numpy)

        return data_numpy, label