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
    preproc_tau_to_crossection, preproc_pow_gptnorm, preproc_pow_gptnorm_reverse,\
    preproc_rrtmgp_inputs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
scale_inputs = False

# Do the outputs need pre-processing, or have they already been scaled (within the Fortran code?)
scale_outputs = True

#  ----------------- File paths -----------------
dat_file = "ml_data_g224_noclouds_CAMS_2011-2013_RFMIPstyle_scaled.nc"     
dat_dir = '/media/peter/samlinux/data/data_training/'

dat_path = dat_dir + dat_file

# LOAD DATA
predictand = 'tau_sw_abs'
x_raw,y_raw,col_dry = load_inp_outp_rrtmgp(dat_path, predictand) # RRTMGP inputs have already been scaled

if scale_inputs:
    x,xmax,xmin = preproc_rrtmgp_inputs(x_raw)
else:
    x = x_raw
    
if scale_outputs:
    # Standardization coefficients loaded from file
    y_mean = ymeans_sw_abs; y_sigma = ysigma_sw_abs
    # Set power scaling coefficient (y == y**(1/nfac))
    nfac = 8 
    
    # Scale by layer number of molecules to obtain absorption cross section
    y   = preproc_tau_to_crossection(y_raw, col_dry)
    # Scale using power-scaling followed by standard-scaling
    y   = preproc_pow_gptnorm(y, nfac, y_mean, y_sigma)
else:
    y = y_raw
    
# Ready for training


import warnings
warnings.filterwarnings("ignore")

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 



mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'
activ       = 'softsign'
fpath       = rootdir+'data/tmp/tmp.h5'
epochs      = 800
patience    = 15
lossfunc    = losses.mean_squared_error
ninputs     = x_tr.shape[1]
ngpt        = y_tr.shape[1]
lr          = 0.001 
batch_size  = 1024

neurons = [16,16]

# batch_size  = 3*batch_size
# lr          = 2 * lr

optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 

# Create model
model = create_model(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')

model.compile(loss=lossfunc, optimizer=optim,
              metrics=mymetrics,  context= ["gpu(0)"])
model.summary()


gc.collect()
# Create earlystopper
earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)

# START TRAINING

history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                    validation_data=(x_val,y_val), callbacks=[earlystopper])
gc.collect()


# y_test_nn       = model.predict(x_test);  
# y_test_nn       = preproc_pow_gptnorm_reverse(y_test_nn, nfac, y_mean, y_sigma)
# tau_test_nn     = y_test_nn * (np.repeat(col_dry_test[:,np.newaxis],ngpt,axis=1))
# plot_hist2d(tau_test,tau_test_nn,20,True)        # 
# plot_hist2d_T(tau_test,tau_test_nn,20,True)      #  

y_nn       = model.predict(x);  
y_nn       = preproc_pow_gptnorm_reverse(y_nn, nfac, y_mean, y_sigma)
y_raw_nn   = y_nn * (np.repeat(col_dry[:,np.newaxis],ngpt,axis=1))

plot_hist2d(y_raw,y_raw_nn,20,True)        # 
plot_hist2d_T(y_raw,y_raw_nn,20,True)      #  