"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RRTMGP (gas optics)

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

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



# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, GAS OPTICS EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
fpath = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2011-2013_noclouds.nc"
fpath_test = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2018_noclouds.nc"

# ----------- config ------------

# Do the inputs need pre-processing? Might have already been scaled (within the Fortran code)
scale_inputs = True

# Do the outputs need pre-processing?
scale_outputs = True

predictand = 'tau_sw_abs'


# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
ml_library = 'pytorch'
# ml_library = 'tf-keras'

# LOAD DATA
# Training + validation data
x_raw,y_raw,col_dry = load_inp_outp_rrtmgp(fpath, predictand) 
# Test data
x_raw_test,y_raw_test,col_dry_test = load_inp_outp_rrtmgp(fpath_test, predictand)

# SCALE DATA
# y coefficients
nfac = 8 # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
# log scaling. Useful when the output is a vector which has a wide range 
# of magnitudes across the vector elements (g-points)
y_mean  = ymeans_sw_abs # standard scaling after square root transformation
y_sigma = ysigma_sw_abs
# x coefficients
xmin = np.array([1.7894626e+02, 2.3025851e+00, 0.0000000e+00, 2.7871470e-04,
       3.8346465e-04, 1.5644504e-07, 0.0000000e+00], dtype=np.float32)
xmax = np.array([3.1476846e+02, 1.1551140e+01, 4.3200806e-01, 5.6353424e-02,
       7.7934266e-04, 3.5097651e-06, 3.3747145e-07], dtype=np.float32) 
xcoeffs = (xmin,xmax)

# Scale data, depending on choices 
x,y             = scale_gasopt(x_raw, y_raw, col_dry, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=y_mean, y_sigma=y_sigma, xcoeffs=xcoeffs)
x_test,y_test   = scale_gasopt(x_raw_test, y_raw_test, col_dry_test, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=y_mean, y_sigma=y_sigma, xcoeffs=xcoeffs)
  
nx = x.shape[1]
ny = y.shape[1] # ngpt


# Split first dataset into training and validation
train_ratio = 0.75
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=1 - train_ratio)



# PYTORCH TRAINING
if (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP#, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    
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

    # Test model
    y_pred = mlp(x_test_torch)
    y_pred = y_pred.detach().numpy()


    np.corrcoef(y_test.flatten(),y_pred.flatten())
    
    y_pred = preproc_pow_gptnorm_reverse(y_pred, nfac, y_mean, y_sigma)
    y_pred = y_pred * (np.repeat(col_dry_test[:,np.newaxis],ny,axis=1))
    
    plot_hist2d(y_raw_test,y_pred,20,True)        # 
    plot_hist2d_T(y_raw,y_pred,20,True)      #  
    
  
# TENSORFLOW-KERAS TRAINING
elif (ml_library=='tf-keras'):
    
    from tensorflow.keras import losses, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    
    import warnings
    warnings.filterwarnings("ignore")
    
    
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    activ       = 'softsign'
    # fpath       = rootdir+'data/tmp/tmp.h5'
    epochs      = 800
    patience    = 15
    lossfunc    = losses.mean_squared_error
    
    lr          = 0.001 
    batch_size  = 1024
    
    neurons = [16,16]
    
    # batch_size  = 3*batch_size
    # lr          = 2 * lr
    
    optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    
    # Create model
    model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ=activ,kernel_init='he_uniform')
    
    model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics, 
                  context= ["gpu(0)"])
    model.summary()
    
    
    gc.collect()
    # Create earlystopper
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    
    # START TRAINING
    
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=[earlystopper])
    gc.collect()
    
    
    y_nn       = model.predict(x);  
    y_nn       = preproc_pow_gptnorm_reverse(y_nn, nfac, y_mean, y_sigma)
    y_raw_nn   = y_nn * (np.repeat(col_dry[:,np.newaxis],ny,axis=1))
    
    plot_hist2d(y_raw,y_raw_nn,20,True)        # 
    plot_hist2d_T(y_raw,y_raw_nn,20,True)      #  
    
    # SAVE MODEL
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # savemodel(kerasfile, model)
    
    # # from keras.models import load_model
    # # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # # model = load_model(kerasfile,compile=False)