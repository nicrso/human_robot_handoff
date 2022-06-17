import os 
import torch 
from torch import nn
import torch.nn.functional as F
from typing import Tuple, List

from losses import DiverseLoss

import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary

class cnn3d(pl.LightningModule):
    """
    3D CNN as described in "VoxNet: A 3D Convolutional Neural Network for Real-Time Object
    Recognition" -- Daniel Maturana and Sebastian Scherer
    """

    def __init__(
        self, 
        in_channels: int = 5,
        out_channels: int = 2,
    ) -> None:
        """
        3D CNN PyTorch Lightning module. 
        :param n_ensemble: Number of models to train 
        :param inplanes: Number of input features
        :param outplanes: Number of output features (classes)
        :param droprate: Probability of dropping a prediction
        """

        super().__init__()

        nc = in_channels

        self.down1 = nn.Sequential(
            self.conv_block(in_channels, nc*4), 
            nn.MaxPool3d(2),
        )
        nc *= 4

        self.down2 = nn.Sequential(
            self.conv_block(nc, nc*4),
            nn.MaxPool3d(2),
        )
        nc *= 4

        self.down3 = nn.Sequential(
            self.conv_block(nc, nc*4),
            nn.MaxPool3d(4),
        )
        nc *= 4

        self.up1 = nn.Sequential(
            self.conv_block(nc, nc//4),
        )
        nc = nc // 4

        self.up2 = nn.Sequential(
            self.conv_block(nc, nc//4),
        )
        nc = nc // 4

        self.up3 = nn.Sequential(
            self.conv_block(nc, nc//4),
        )
        nc = nc // 4

        self.up4 = nn.Conv3d(nc, out_channels, kernel_size=3, padding=1)

        #Look into why this is done in Voxnet paper 
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Repeatable convolutional block. 
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :return: The sequential convolutional block
        """

        return nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor, c) -> torch.Tensor:
        """
        Forward defines prediction/inference actions.
        :param x: Input tensor 
        :param c:
        :return: Model output tensor
        """
        #lightning: forward defines prediction/inference actions
        if self.droprate > 0:
            x = self.drop(x)

        #Encoding pass
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        #Decoding pass
        x = x.view(x.shape[0], -1, 4, 4, 4)
        x = F.interpolate(self.up1(x), scale_factor=4)
        x = F.interpolate(self.up2(x), scale_factor=2)
        x = F.interpolate(self.up3(x), scale_factor=2)
        x = self.up4(x)

        return x

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int,
    ) -> float:
        """
        Returns the training loss and logs loss.
        :param batch: The input and target training data
        :param batch_idx: The index of the given batch
        :return: The training loss
        """

        geom, tex_targs = batch 
        tex_preds = self(geom)

        loss, _ = self.train_loss_fn(tex_preds, tex_targs)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        idx: int,
    ) -> float:
        """
        Returns the validation loss. 
        :param batch: The input and target validation data
        :param idx: Current batch index
        :return: The validation loss
        """

        geom, tex_targs = batch 
        tex_preds = self(geom)

        loss, _ = self.val_loss_fn(tex_preds, tex_targs)

        self.log(loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures optimizers for the model.
        :return: Configured optimizers 
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #Note: might also try SGD, but research shows fine-tuned adam outperforms 


        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        return [optimizer], [lr_scheduler]

if __name__ == "__main__":
    model = cnn3d()
    print(model) 