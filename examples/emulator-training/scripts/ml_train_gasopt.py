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
    preproc_pow_gptnorm_reverse,scale_gasopt
from ml_eval_funcs import plot_hist2d, plot_hist2d_T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import pytorch_lightning as pl

from keras import losses


class MLP(pl.LightningModule):
  
  def __init__(self, nx, ny):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(nx, 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, ny)
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


# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, GAS OPTICS EMULATION  ------------------------
# ----------------------------------------------------------------------------

# Do the inputs need pre-processing, or have they already been scaled (within the Fortran code)?
scale_inputs = True

# Do the outputs need pre-processing, or have they already been scaled (within the Fortran code?)
scale_outputs = True

#  ----------------- File paths -----------------
dat_file = "ml_data_g224_noclouds_CAMS_2011-2013_RFMIPstyle_scaled.nc"     
dat_dir = '/media/peter/samlinux/data/data_training/'

dat_file = "ml_data_g224_CAMS_2018_clouds.nc"
dat_dir  = '/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/'

dat_path = dat_dir + dat_file

# LOAD DATA
predictand = 'tau_sw_abs'
x_raw,y_raw,col_dry = load_inp_outp_rrtmgp(dat_path, predictand) # RRTMGP inputs have already been scaled

# SCALE DATA
ymean  = ymeans_sw_abs
ysigma = ysigma_sw_abs

xmin = np.array([1.7894626e+02, 2.3025851e+00, 0.0000000e+00, 2.7871470e-04,
       3.8346465e-04, 1.5644504e-07, 0.0000000e+00], dtype=np.float32)
xmax = np.array([3.1476846e+02, 1.1551140e+01, 4.3200806e-01, 5.6353424e-02,
       7.7934266e-04, 3.5097651e-06, 3.3747145e-07], dtype=np.float32) 
xcoeffs = (xmin,xmax)

x,y =  scale_gasopt(x_raw, y_raw, col_dry, scale_inputs, scale_outputs, 
                    ymean, ysigma, xcoeffs=xcoeffs)
        
nx = x.shape[1]; ny = y.shape[1]

train_ratio = 0.75
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=1 - train_ratio)


# LOAD AND SCALE TEST DATA
dat_file_test = "ml_data_g224_CAMS_2018_noclouds.nc"
dat_path_test = dat_dir + dat_file_test

x_raw_test,y_raw_test,col_dry_test = load_inp_outp_rrtmgp(dat_path_test, predictand)

x_test,y_test =  scale_gasopt(x_raw_test, y_raw_test, col_dry_test, scale_inputs, scale_outputs, 
                    ymean, ysigma, xcoeffs=xcoeffs)


batch_size = 256

x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)

x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
data_val    = TensorDataset(x_val_torch,y_val_torch)

x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
data_test    = TensorDataset(x_test_torch,y_test_torch)

mlp = MLP(nx=nx,ny=ny)
#trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)

trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=100)
#trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=5,num_processes=3)


trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
            val_dataloaders=DataLoader(data_val,batch_size=batch_size))



#
#trainer.predict(mlp,dataloaders=data_val)
nfac = 8

ns_test = x_test.shape[0]
ysc_test_pred = mlp.predict_step(x_test_torch,batch_idx=np.arange(0,ns_test))
ysc_test_pred = ysc_test_pred.detach().numpy()


np.corrcoef(y_test.flatten(),ysc_test_pred.flatten())

y_test_pred = preproc_pow_gptnorm_reverse(ysc_test_pred, nfac, ymean, ysigma)
y_test_pred = y_test_pred * (np.repeat(col_dry_test[:,np.newaxis],ny,axis=1))

plot_hist2d(y_raw_test,y_test_pred,20,True)        # 
plot_hist2d_T(y_raw,y_test_pred,20,True)      #  




# Old keras code

#
#import warnings
#warnings.filterwarnings("ignore")
#

#
#
#mymetrics   = ['mean_absolute_error']
#valfunc     = 'val_mean_absolute_error'
#activ       = 'softsign'
#fpath       = rootdir+'data/tmp/tmp.h5'
#epochs      = 800
#patience    = 15
#lossfunc    = losses.mean_squared_error
#
#lr          = 0.001 
#batch_size  = 1024
#
#neurons = [16,16]
#
## batch_size  = 3*batch_size
## lr          = 2 * lr
#
#optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
#
## Create model
#model = create_model(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')
#
#model.compile(loss=lossfunc, optimizer=optim,
#              metrics=mymetrics,ngpt  context= ["gpu(0)"])
#model.summary()
#
#
#gc.collect()
## Create earlystopper
#earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
#
## START TRAINING
#
#history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
#                    validation_data=(x_val,y_val), callbacks=[earlystopper])
#gc.collect()
#
#
## y_test_nn       = model.predict(x_test);  
## y_test_nn       = preproc_pow_gptnorm_reverse(y_test_nn, nfac, y_mean, y_sigma)
## tau_test_nn     = y_test_nn * (np.repeat(col_dry_test[:,np.newaxis],ngpt,axis=1))
## plot_hist2d(tau_test,tau_test_nn,20,True)        # 
## plot_hist2d_T(tau_test,tau_test_nn,20,True)      #  
#
#
#y_nn       = model.predict(x);  
#y_nn       = preproc_pow_gptnorm_reverse(y_nn, nfac, y_mean, y_sigma)
#y_raw_nn   = y_nn * (np.repeat(col_dry[:,np.newaxis],ngpt,axis=1))
#
#plot_hist2d(y_raw,y_raw_nn,20,True)        # 
#plot_hist2d_T(y_raw,y_raw_nn,20,True)      #  