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

from ml_loaddata import ymeans_sw_abs, ysigma_sw_abs, ymeans_sw_ray, ysigma_sw_ray, \
    load_rrtmgp, preproc_pow_standardization_reverse,scale_gasopt
from ml_eval_funcs import plot_hist2d, plot_hist2d_T
import matplotlib.pyplot as plt




# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, GAS OPTICS EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
datadir     = "/home/peter/data/"

fpath       = datadir+"ml_data_g224_CAMS_2012-2016_noclouds.nc"
fpath_val   = datadir+"ml_data_g224_CAMS_2017_noclouds.nc"
fpath_test  = datadir+"ml_data_g224_CAMS_2018_noclouds.nc"

fpath       = datadir+"RRTMGP_data_g224_CAMS_2009-2018_sans_2014-2015_RND.nc"
fpath_val   = datadir+"RRTMGP_data_g224_CAMS_2014_RND.nc"
fpath_test  = datadir+"RRTMGP_data_g224_CAMS_2015_RND.nc"


# Just one dataset
# fpath_val = None
# fpath_test = None

# ----------- config ------------

# Do the inputs need pre-processing? Might have already been scaled (within the Fortran code)
scale_inputs = True

# Do the outputs need pre-processing?
scale_outputs = True

# Choose one of the following predictands (target output)
# 'tau_lw', 'planck_frac', 'tau_sw_abs', 'tau_sw_ray', 'tau_sw', 'ssa_sw'
predictand = 'tau_sw_abs'
predictand = 'tau_sw_ray'

# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
ml_library = 'tf-keras'
# ml_library = 'tf-keras'

# Model training: use CPU or GPU?
use_gpu = False

retrain_mae = False

# ----------- config ------------

# LOAD DATA given three separate datasets for training - validation - testing
# Training data
x_tr_raw,y_tr_raw,col_dry_tr        = load_rrtmgp(fpath, predictand) 

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw, y_val_raw, col_dry_val   = load_rrtmgp(fpath_val, predictand)
    x_test_raw,y_test_raw,col_dry_test  = load_rrtmgp(fpath_test, predictand)
else: # if we only have one dataset, split manually
    from sklearn.model_selection import train_test_split
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    testval_ratio = test_ratio/(test_ratio + validation_ratio)
    # first split into two, training and test+val
    x_tr_raw, x_test_raw, y_tr_raw, y_test_raw = \
        train_test_split(x_tr_raw, y_tr_raw, test_size=1-train_ratio)
    # then split the latter into to testing and val
    x_val_raw, x_test_raw, y_val_raw, y_test_raw = \
        train_test_split(x_test_raw, y_test_raw, test_size=testval_ratio) 



# SCALE DATA
# y coefficients
nfac = 8 # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
# log scaling. Useful when the output is a vector which has a wide range 
# of magnitudes across the vector elements (g-points)
# standard scaling after square root transformation
if (predictand == 'tau_sw_abs'):
    y_mean  = ymeans_sw_abs 
    y_sigma = ysigma_sw_abs
elif (predictand == 'tau_sw_ray'):
    y_mean  = ymeans_sw_ray #
    y_sigma = ysigma_sw_ray
else: 
    print("SPECIFY Y SCALING COEFFICIENTS")
    
    
    
    

    
# INPUT SCALING

#  tlay play h2o o3 co2 ch4 n2o
# xmin = np.array([1.60E2, 5.15E-3, 1.01E-2, 4.36E-3,1.41E-4, 2.55E-8, 0.00E0], dtype=np.float32)
# xmax = np.array([ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3,
#          3.6000001E-6, 5.8135214E-7], dtype=np.float32) 

#  tlay play h2o o3 co2 n2o ch4
xmin = np.array([1.60E2, 5.15E-3, 1.01E-2, 4.36E-3, 1.41E-4, 0.00E0, 2.55E-8], dtype=np.float32)
xmax = np.array([ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3,
         5.8135214E-7, 3.6000001E-6], dtype=np.float32) 
xcoeffs = (xmin,xmax)

# Scale data, depending on choices 
x_tr,y_tr       = scale_gasopt(x_tr_raw, y_tr_raw, col_dry_tr, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=y_mean, y_sigma=y_sigma, xcoeffs=xcoeffs)
# val
x_val,y_val     = scale_gasopt(x_val_raw, y_val_raw, col_dry_val, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=y_mean, y_sigma=y_sigma, xcoeffs=xcoeffs)
# test
x_test,y_test   = scale_gasopt(x_test_raw, y_test_raw, col_dry_test, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=y_mean, y_sigma=y_sigma, xcoeffs=xcoeffs)

x_tr[x_tr<0.0] = 0.0
x_val[x_val<0.0] = 0.0
x_test[x_test<0.0] = 0.0

  
nx = x_tr.shape[1]
ny = y_tr.shape[1] # = number of g-points

# Split first dataset into training and validation
# train_ratio = 0.75
# x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=1 - train_ratio)


# PYTORCH TRAINING
if (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP#, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    lr          = 0.001
    batch_size  = 512
    nneur       = 16
    mymodel = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.Softsign(), # first hidden layer
          nn.Linear(nneur, nneur),
          nn.Softsign(), # second hidden layer
          nn.Linear(nneur, ny) # output layer
        )
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)
    
    x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
    data_test    = TensorDataset(x_test_torch,y_test_torch)
    
    mlp = MLP(nx=nx,ny=ny,learning_rate=lr,SequentialModel=mymodel)


    mc = pl.callbacks.ModelCheckpoint(monitor='val_loss',every_n_epochs=2)
    
    if use_gpu:
        trainer = pl.Trainer(gpus=1, deterministic=True)
    else:
        num_cpu_threads = 8
        trainer = pl.Trainer(accelerator="ddp_cpu", callbacks=[mc], deterministic=True,
                num_processes=  num_cpu_threads) 
                #plugins=pl.plugins.DDPPlugin(find_unused_parameters=False))
    
    # START TRAINING
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
            val_dataloaders=DataLoader(data_val,batch_size=batch_size))

    # PREDICT OUTPUTS FOR TEST DATA
    def eval_valdata():
        y_pred = mlp(x_val_torch)
        y_pred = y_pred.detach().numpy()
    
        # np.corrcoef(y_test.flatten(),y_pred.flatten())
        
        y_pred = preproc_pow_standardization_reverse(y_pred, nfac, y_mean, y_sigma)
        if predictand not in ['planck_frac', 'ssa_sw']:
            y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))
        
        plot_hist2d(y_val_raw,y_pred,20,True)   # 
        plot_hist2d_T(y_val_raw,y_pred,20,True)      # Transmittance  
    
    eval_valdata()
    
  
# TENSORFLOW-KERAS TRAINING
elif (ml_library=='tf-keras'):
    import tensorflow as tf
    from tensorflow.keras import losses, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    
    # Model architecture
    # First hidden layer (input layer) activation
    activ0      = 'softsign'
    # activ0       = 'relu'
    # Activation in other hidden layers
    activ       =  activ0    
    # Activation for last layer
    activ_last   = 'linear'
    # Number of neurons in each hidden layer
    neurons     = [16,16]
    
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    # fpath       = rootdir+'data/tmp/tmp.h5'
    epochs      = 800
    patience    = 15
    lossfunc    = losses.mean_squared_error
    lr          = 0.001 
    batch_size  = 1024
    neurons     = [16,16]
    
    batch_size  = 4096
    lr          = 0.01
    
    # batch_size  = 3*batch_size
    # lr          = 2 * lr
    
    if use_gpu:
        devstr = '/gpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    else:
        num_cpu_threads = 12
        devstr = '/cpu:0'
        # Maximum number of threads to use for OpenMP parallel regions.
        os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
        # Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpu_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(1)
        os.environ['KMP_BLOCKTIME'] = '1' 

        tf.config.threading.set_intra_op_parallelism_threads(
            num_cpu_threads
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            1
        )
        tf.config.set_soft_device_placement(True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    optim = optimizers.Adam(lr=lr)

    # Create and compile model
    # model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
    #                          activ_last = activ_last, kernel_init='he_uniform')
    model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                             activ_last = activ_last, kernel_init='lecun_uniform')
    model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
    model.summary()

    # Create earlystopper and possibly other callbacks
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    callbacks = [earlystopper]

    # START TRAINING
    with tf.device(devstr):
        history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                            validation_data=(x_val,y_val), callbacks=callbacks)    
        
    if retrain_mae:
        model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
        callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)]
        with tf.device(devstr):
            history2 = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                                validation_data=(x_val,y_val), callbacks=callbacks)
            
        
    # PREDICT OUTPUTS FOR TEST DATA
    def eval_valdata():
        y_pred       = model.predict(x_val);  
        y_pred       = preproc_pow_standardization_reverse(y_pred, nfac, y_mean, y_sigma)
        
        if predictand not in ['planck_frac', 'ssa_sw']:
            y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))
            
        plot_hist2d(y_val_raw,  y_pred,20,True)   # 
        plot_hist2d_T(y_val_raw,y_pred,20,True)      # Transmittance  
        
    eval_valdata()

    # SAVE MODEL
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16-CAMS-NEW-mae.h5"
    # savemodel(kerasfile, model)
    
    # from keras.models import load_model
    # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # model = load_model(kerasfile,compile=False)