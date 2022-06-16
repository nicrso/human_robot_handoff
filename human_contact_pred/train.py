
from asyncio.log import logger
import os 
import configparser
import argparse

import torch 
from torch import embedding, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from voxel_dataset import VoxelDataset
from model import DiverseVoxNet, VoxNet

osp = os.path

 
def train(
    data_dir, 
    instruction, 
    config_file, 
    checkpoint_file=None,
    experiment_suffix=None,
    include_sessions=None,
    exclude_sessions=None
):

    #Config
    config = configparser.ConfigParser()
    config.read(config_file)

    section = config['optim']
    batch_size = section.getint('batch_size')
    max_epochs = section.getint('max_epochs')
    val_interval = section.getint('val_interval')
    do_val = val_interval > 0
    base_lr = section.getfloat('base_lr')
    momentum = section.getfloat('momentum')
    weight_decay = section.getfloat('weight_decay')

    section = config['misc']
    log_interval = section.getint('log_interval')
    shuffle = section.getboolean('shuffle')
    num_workers = section.getint('num_workers')

    section = config['hyperparams']
    n_ensemble = section.getint('n_ensemble')
    diverse_beta = section.getfloat('diverse_beta')
    pos_weight = section.getfloat('pos_weight')
    droprate = section.getfloat('droprate')
    lr_step_size = section.getint('lr_step_size', 10000)
    lr_gamma = section.getfloat('lr_gamma', 1.0)
    grid_size = section.getint('grid_size')
    random_rotation = section.getfloat('random_rotation')

    resume = False if checkpoint_file is None else True

    kwargs = dict(
        data_dir=data_dir, 
        instruction=instruction,
        include_sessions=include_sessions, 
        exclude_sessions=exclude_sessions,
        n_ensemble=n_ensemble
        )

    #Training dataset
    train_dset = VoxelDataset(
        grid_size=grid_size,
        random_rotation=random_rotation,
        is_train=True,
        **kwargs
    )
    train_loader = DataLoader(
        train_dset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


    #Val dataset
    val_dset = VoxelDataset(
        grid_size=grid_size,
        random_rotation=0,
        is_train=False,
        **kwargs
    )
    val_loader = DataLoader(
        val_dset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    #Create model
    voxnet = DiverseVoxNet(
        n_ensemble=n_ensemble,
        droprate=droprate
    )

    #Lightning training
    logger = TensorBoardLogger("tb_logs", name="voxnet")
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger)

    if resume:
        trainer.fit(
            model=voxnet, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=checkpoint_file
        )
    else: 
        trainer.fit(
            model=voxnet, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
        default=osp.join('data', 'voxelized_meshes'))
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--suffix', default=None)
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument('--include_sessions', default=None)
    parser.add_argument('--exclude_sessions', default=None)
    args = parser.parse_args()

    include_sessions = None
    if args.include_sessions is not None:
        include_sessions = args.include_sessions.split(',')
    exclude_sessions = None
    if args.exclude_sessions is not None:
        exclude_sessions = args.exclude_sessions.split(',')
    train(osp.expanduser(args.data_dir), args.instruction, args.config_file,
        experiment_suffix=args.suffix,checkpoint_file=args.checkpoint_file, include_sessions=include_sessions,
        exclude_sessions=exclude_sessions)