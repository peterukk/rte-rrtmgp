#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE (the solver)

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import gc
import numpy as np
from ml_loaddata import load_inp_outp_rte_sw,  preproc_minmax_inputs
from ml_eval_funcs import plot_hist2d

def broadband_reduce(y):
    ns,ny = y.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.reshape(y,(ncol,ngpt,ny))
    return np.sum(y,axis=1)

def scale_outputs_gpt(y_raw):
    ns,ny = y_raw.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y_sc = np.copy(y_raw)
    y_sc = np.reshape(y_sc,(ncol,ngpt,ny))
    
    y_mean = np.zeros(ngpt)
    for igpt in range(ngpt):
        y_mean[igpt] = np.mean(y_sc[:,igpt,:])
        
    y_sigma = np.repeat(y_sc.std(),ngpt)  # 467.72
    
    for igpt in range(ngpt):
        y_sc[:,igpt,:] = (y_sc[:,igpt,:] - y_mean[igpt]) / y_sigma[igpt]

    y_sc = np.reshape(y_sc,(ns,ny))
    return y_sc, y_mean, y_sigma

def reverse_scale_outputs_gpt(y_sc, y_mean, y_sigma):
    ns,ny = y_sc.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.copy(y_sc)
    y = np.reshape(y,(ncol,ngpt,ny))
    
    for igpt in range(ngpt):
        y[:,igpt,:] = y[:,igpt,:]*y_sigma[igpt] + y_mean[igpt]

    y = np.reshape(y,(ns,ny))
    return y

def scale_outputs_gpt_lay(y_raw, y_coeffs=None):
    ns,ny = y_raw.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.copy(y_raw)
    y = np.reshape(y,(ncol,ngpt*ny))
    coeffs_provided = True
    
    ny2 = y.shape[1]
    if (y_coeffs == None):
        coeffs_provided = False
        y_sigma = np.repeat(y.std(),ny2)  # 467.72    
        y_mean = np.zeros(ny2)
        for i in range(ny2):
            y_mean[i] = np.mean(y[:,i])
    else:
        y_mean, y_sigma = y_coeffs
    for i in range(ny2):
        y[:,i] = (y[:,i] - y_mean[i]) / y_sigma[i]

    y = np.reshape(y,(ns,ny))
    if coeffs_provided:
        return y
    else:
        return y, y_mean, y_sigma

def reverse_scale_outputs_gpt_lay(y_sc, y_mean, y_sigma):
    ns,ny = y_sc.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.copy(y_sc)
    y = np.reshape(y,(ncol,ngpt*ny))
    
    for i in range(y.shape[1]):
        y[:,i] = y[:,i]*y_sigma[i] + y_mean[i]

    y = np.reshape(y,(ns,ny))
    return y
# ----------------------------------------------------------------------------


#  ----------------- File paths -----------------
fpath_tr    = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2012-2016_clouds.nc"
fpath_val   = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2017_clouds.nc"
fpath_test  = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2018_clouds.nc"

fpath       = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2011-2013_clouds.nc"
fpath_val   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"
fpath_test   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"


# ----------- config ------------

scale_inputs    = True
scale_outputs   = True

# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
# ml_library = 'pytorch'
ml_library = 'tf-keras'

# Model training: use GPU or CPU?
use_gpu = False

# ----------- config ------------

# Load data
x_tr_raw, y_tr_raw, y_tr_bb = load_inp_outp_rte_sw(fpath)
# y_tr_bb is broadband flux

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw, y_val_raw, y_val_bb   = load_inp_outp_rte_sw(fpath_val)
    x_test_raw,y_test_raw,y_test_bb  = load_inp_outp_rte_sw(fpath_test)
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


if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    
    x_tr, xmin,xmax = preproc_minmax_inputs(x_tr_raw)
    x_val           = preproc_minmax_inputs(x_val_raw,  (xmin,xmax)) 
    x_test          = preproc_minmax_inputs(x_test_raw, (xmin,xmax)) 
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
if scale_outputs: 
    
    # Scale by GPT+LAY 
    y_tr, y_mean, y_sigma = scale_outputs_gpt_lay(y_tr_raw)
    
    y_val  = scale_outputs_gpt_lay(y_val_raw, (y_mean, y_sigma))
    y_test = scale_outputs_gpt_lay(y_test_raw, (y_mean, y_sigma))

    # Scale by GPT
    # y,y_mean,y_sigma = scale_outputs_gpt(y_raw)
    
    # Original
    # ny = y_raw.shape[1]      # y.shape (14400, 122)
    # nobs = y_raw.shape[0]
    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = y_raw[:,i].mean()
    #     # y_sigma[i] = y_raw[:,i].std()
    # # y_mean = np.repeat(y_raw.mean(),ny)
    # y_sigma = np.repeat(y_raw.std(),ny)  # 467.72

    # nfac = 1
    # y  = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw

gc.collect()
# Number of inputs and outputs    
nx = x_tr_raw.shape[1]
ny = y_tr_raw.shape[1]    
# Ready for training



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
    nneur       = 160
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
        trainer = pl.Trainer(gpus=0, deterministic=True)
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
        y_pred    = reverse_scale_outputs_gpt_lay(y_pred,y_mean,y_sigma)
        y_pred_bb = broadband_reduce(y_pred) 
        plot_hist2d(y_test_bb,y_pred_bb,20,True)      #  0.979
    
    eval_valdata()

  
# TENSORFLOW-KERAS TRAINING
elif (ml_library=='tf-keras'):
    import tensorflow as tf
    from tensorflow.keras import losses, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    
    # Model architecture
    # First hidden layer (input layer) activation
    activ0      = 'softsign'
    # activ0       = 'relu'
    # Activation in other hidden layers
    activ       =  activ0    
    # Activation for last layer
    activ_last   = 'linear'
    

    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    ninputs     = x_tr.shape[1]
    lr          = 0.001
    # lr          = 0.0001 
    # lr          = 0.0002 
    batch_size  = 256
    batch_size  = 512
    neurons     = [182, 182]
    
    if use_gpu:
        devstr = '/gpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    else:
        num_cpu_threads = 4
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
        
    # TEST
    y_pred    = model.predict(x_test);  
    y_pred    = reverse_scale_outputs_gpt_lay(y_pred,y_mean,y_sigma)
    y_pred_bb = broadband_reduce(y_pred) 
    plot_hist2d(y_test_bb,y_pred_bb,20,True)      #  0.979
    
    diff = np.abs(y_test_bb - y_pred_bb)
    print("max diff {}".format(np.max(diff)))
