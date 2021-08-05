#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of 
RRTMGP gas optics scheme

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, 
scales the data if requested, and trains a neural network. 

Alternatively, an automatic tuning method can be used for
finding a good set of hyperparameters (expensive).

Right now just a placeholder, pasted some of the code I used in my paper

Contributions welcome!

@author: Peter Ukkonen
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import pytorch_lightning as pl
# from pytorch_lightning.plugins import DeepSpeedPlugin
# from deepspeed.ops.adam import DeepSpeedCPUAdam

class MLP(pl.LightningModule):
  
  def __init__(self, nx, ny, nneur=8, learning_rate=None, SequentialModel=None):
    super().__init__()
    if SequentialModel==None:
        self.layers = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.ReLU(),
          nn.Linear(nneur, nneur),
          nn.ReLU(),
          nn.Linear(nneur, ny)
        )
    else:
        self.layers = SequentialModel
    self.mse = nn.MSELoss()
    # Hyperparameters
    self.learning_rate = learning_rate
    self.save_hyperparameters()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.mse(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.mse(y_hat, y)
    self.log('val_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer


# class MLP_cpu(pl.LightningModule):    
#     from pytorch_lightning.plugins import DeepSpeedPlugin
#     from deepspeed.ops.adam import DeepSpeedCPUAdam
      
#     def __init__(self, nx, ny, nneur=8, learning_rate=None, SequentialModel=None):
#       super().__init__()
#       if SequentialModel==None:
#           self.layers = nn.Sequential(
#             nn.Linear(nx, nneur),
#             nn.ReLU(),
#             nn.Linear(nneur, nneur),
#             nn.ReLU(),
#             nn.Linear(nneur, ny)
#           )
#       else:
#           self.layers = SequentialModel
#       self.mse = nn.MSELoss()
#       # Hyperparameters
#       self.learning_rate = learning_rate
#       self.save_hyperparameters()
      
#     def forward(self, x):
#       return self.layers(x)
    
#     def training_step(self, batch, batch_idx):
#       x, y = batch
#       x = x.view(x.size(0), -1)
#       y_hat = self.layers(x)
#       loss = self.mse(y_hat, y)
#       self.log('train_loss', loss)
#       return loss
  
#     def validation_step(self, batch, batch_idx):
#       x, y = batch
#       x = x.view(x.size(0), -1)
#       y_hat = self.layers(x)
#       loss = self.mse(y_hat, y)
#       self.log('val_loss', loss)
#       return loss
    
#     def configure_optimizers(self):
#       optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
#       return optimizer
