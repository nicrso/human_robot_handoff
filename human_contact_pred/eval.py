
import os 
import configparser
import argparse

import torch 
from torch import embedding, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from voxel_dataset import VoxelDataset
from model import DiverseVoxNet, VoxNet

osp = os.path

#load model from checkpoint
path_to_checkpoint = "."
model = DiverseVoxNet.load_from_checkpoint(path_to_checkpoint)

#print model hyperparams
checkpoint = torch.load(path_to_checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])

#disable randomness, dropout, etc. 
model.eval()

#predict with the model