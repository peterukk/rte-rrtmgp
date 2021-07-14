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
import os
import gc
import numpy as np

from ml_loaddata import ymeans_sw_abs, ysigma_sw_abs, load_inp_outp_rrtmgp, \
    preproc_tau_to_crossection, preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from sklearn.model_selection import train_test_split

#import torch
#from torch import nn
#from torchvision.datasets import CIFAR10
#from torch.utils.data import DataLoader
#from torchvision import transforms
#import pytorch_lightning as pl
#
#class MLP(pl.LightningModule, nfeatures, ngpt):
#  
#  def __init__(self):
#    super().__init__()
#    self.layers = nn.Sequential(
#      nn.Linear(nfeatures, 64),
#      nn.ReLU(),
#      nn.Linear(64, 64),
#      nn.ReLU(),
#      nn.Linear(32, ngpt)
#    )
#    self.mse = nn.MSELoss()
#    
#  def forward(self, x):
#    return self.layers(x)
#  
#  def training_step(self, batch, batch_idx):
#    x, y = batch
#    x = x.view(x.size(0), -1)
#    y_hat = self.layers(x)
#    loss = self.mse(y_hat, y)
#    self.log('train_loss', loss)
#    return loss
#  
#  def configure_optimizers(self):
#    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
#    return optimizer



# Okay, I really need to switch to OOP in my Python code.
# To take this train program as an example, it could use OOP (classes/functions) for
# - input scaling 
#  --- takes data, optional min max values (if not uses data values)
#  --- power scaling yes/no for set inputs? use some criteria?
#  ---- returns scaled outputs, coefficients
# - output scaling
# ---- takes data, optional coefficients, or uses data values
# ---- specify method: regular standard scaling or alternative
# ---- returns scaled inputs, coefficients
#- 


# ----------- LOAD DATA --------------------
#
#
#if __name__ == "__main__":
#    
#    if len(sys.argv) != 7:
#	print('Usage:')
#	print(f'python {sys.argv[0]} explore <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <database path>')
#	print(f'python {sys.argv[0]} train <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <model path>')
#	sys.exit(1)
#    
#    mode = sys.argv[1]
#    
#    x_training_path = sys.argv[2]
#    y_training_path = sys.argv[3]
#    
#    x_validation_path = sys.argv[4]
#    y_validation_path = sys.argv[5]
#        
#    fpath_input  = "home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_ml_input_output/CAMS_2011_noclouds"
#    fpath_output = "home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_ml_input_output/CAMS_2011_noclouds"
#
#
#    # GET DATA
#    train_images = np.load('./data/image_data.npy')
#    train_labels = np.load('./data/image_labels2.npy')
#    test_images = train_images
#    test_labels = train_labels



# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, GAS OPTICS EMULATION  ------------------------
# ----------------------------------------------------------------------------

# Do the inputs need pre-processing, or have they already been scaled (within the Fortran code)?
scale_inputs = True

# Do the outputs need pre-processing, or have they already been scaled (within the Fortran code?)
scale_outputs = True

#  ----------------- File paths -----------------
dat_file = "ml_data_g224_clouds_CAMS_2011_RFMIPstyle.nc"                                
# this_dir = ""
# emulator_dir  = this_dir + "../"
# dat_path  = emulator_dir + "data_training/" + dat_file
dat_path = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/" + dat_file

# LOAD DATA
predictand = 'tau_sw_abs'
x_raw,y_raw,col_dry = load_inp_outp_rrtmgp(dat_path, predictand) # RRTMGP inputs have already been scaled

if scale_inputs:
    print("call input scaling function here")
else:
    x = x_raw
    
if scale_outputs:
    # Standardization coefficients loaded from file
    y_mean = ymeans_sw_abs; y_sigma = ysigma_sw_abs
    # Set power scaling coefficient (y == y**(1/nfac))
    nfac = 8 
    
    # Scale by layer number of molecules to obtain absorption cross section
    y_raw   = preproc_tau_to_crossection(y_raw, col_dry)
    # Scale using power-scaling followed by standard-scaling
    y       = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
else:
    y = y_raw
    
# Ready for training
