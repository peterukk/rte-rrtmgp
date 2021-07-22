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

class MLP(pl.LightningModule):
  
  def __init__(self, nx, ny, nneur):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(nx, nneur),
      nn.ReLU(),
      nn.Linear(nneur, nneur),
      nn.ReLU(),
      nn.Linear(nneur, ny)
    )
    self.mse = nn.MSELoss()
    
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
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer

