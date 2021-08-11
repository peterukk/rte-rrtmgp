#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating reflectance-transmittance computations in RTE

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
import os,gc
import numpy as np

from ml_loaddata import load_inp_outp_reftrans, preproc_minmax_inputs, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse, gen_synthetic_inp_outp_reftrans
from ml_eval_funcs import plot_hist2d

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# --- TRAIN NEURAL NETS TO REPLACE REFLECTANCE-TRANSMITTANCE COMPUTATIONS ---
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
# fpath = "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc"  
# fpath_rfmip = "/media/peter/samlinux/data/data_training/ml_data_reftrans_RFMIP.nc"
# fpath  ='/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc'

# fpath_tr    = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2012-2016_clouds.nc"
# fpath_val   = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2017_clouds.nc"
# fpath_test  = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2018_clouds.nc"

fpath_tr    = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2011-2013_clouds.nc"
fpath_val   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"
fpath_test  = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_RFMIP_noclouds.nc"

# ----------- config ------------
scale_inputs = True
scale_outputs = True

# CAMS data: balance samples so that half of them correspond to cloudy layers?
# This is done by extracting all cloudy layers, and then an equal amount of 
# random non-cloudy layers
balance_samples = True

# Generate data synthetically by doing hypercube sampling of inputs and 
# generating corresponding outputs on the fly? For REFTRANS computations this 
# is very doable because there's only 4 inputs; additionally the 
# reftrans routine can easily be coded in Python
synthetic_data_supplement = True

# Add no-scattering transmittance as a NN input?
add_Tnoscat = True

# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
# ml_library = 'pytorch'
ml_library = 'tf-keras'

# Data visualization: plot distributions of input and output data
plot_distributions = False

# Model training: use CPU or GPU?
use_gpu = False

# Model evaluation: plot scatterplots of individual outputs
plot_eval = False

# ----------- config ------------


# LOAD DATA
x_tr_raw,   y_tr_raw    = load_inp_outp_reftrans(fpath_tr, balance_samples)

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw,  y_val_raw   = load_inp_outp_reftrans(fpath_val, balance_samples)
    x_test_raw, y_test_raw  = load_inp_outp_reftrans(fpath_test)
    
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
        
        
# this data is too large for reftrans emulation since it's the number of samples
# is ngpt * nlay * ncol = 224 * 60 * ncol, which easily reaches tens of millions
# with CAMS data

# lets extract random samples
if balance_samples:
    frac = 0.08
else:
    frac = 0.03

nrows       = x_tr_raw.shape[0]
inds_rand   = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
x_tr_raw = x_tr_raw[inds_rand,:]; y_tr_raw = y_tr_raw[inds_rand,:]
nrows       = x_val_raw.shape[0]
inds_rand   = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
x_val_raw   = x_val_raw[inds_rand,:]; y_val_raw = y_val_raw[inds_rand,:]
nrows       = x_test_raw.shape[0]
inds_rand   = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
x_test_raw  = x_test_raw[inds_rand,:]; y_test_raw = y_test_raw[inds_rand,:]

print( "{:e} training samples remain after trimming".format(x_tr_raw.shape[0]))


if synthetic_data_supplement:
    minmax_ssa  = (0.0, 1.0)
    minmax_g    = (0.0, 0.55)
    minmax_mu0  = (1e-3, 1.0)

    # The observed distribution has mostly small tau values
    minmax_tau  = (0.1, 20.0)
    nsamples    = np.int(5e5)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # Clear-sky conditions: g is zero
    minmax_tau  = (1e-09, 120000.00)
    nsamples = np.int(5e5)
    minmax_g    = None
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # larger g values
    minmax_ssa  = (0.35, 1.0)
    minmax_tau  = (1e-3, 100.0)
    minmax_g    = (0.4, 0.8)
    nsamples    = np.int(5e5)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # EVEN LARGER g values
    minmax_ssa  = (0.9, 1.0)
    minmax_tau  = (0.05, 3.0)
    minmax_g    = (0.6, 0.9)
    minmax_mu0  = (0.9, 1.0)

    nsamples    = np.int(5e5)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    print( "{:e} training samples after adding synthetic data".format(x_tr_raw.shape[0]))

# Add Tnoscat as input if requested
if add_Tnoscat:
    tnoscat = np.exp(-x_tr_raw[:,0]*(1/x_tr_raw[:,3]))
    x_tr_raw = np.hstack((x_tr_raw,np.reshape(tnoscat,(x_tr_raw.shape[0],1))))
    
    tnoscat = np.exp(-x_val_raw[:,0]*(1/x_val_raw[:,3]))
    x_val_raw = np.hstack((x_val_raw,np.reshape(tnoscat,(x_val_raw.shape[0],1))))
    
    tnoscat = np.exp(-x_test_raw[:,0]* (1/x_test_raw[:,3]))
    x_test_raw = np.hstack((x_test_raw,np.reshape(tnoscat,(x_test_raw.shape[0],1)))) 
    
# Number of inputs and outputs    
nx = x_tr_raw.shape[1]
ny = y_tr_raw.shape[1]    
    
# Ensure outputs are positive 
y_tr_raw[y_tr_raw<0.0] = 0.0
y_val_raw[y_val_raw<0.0] = 0.0
y_test_raw[y_test_raw<0.0] = 0.0


if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    
    # Square-root scaling of optical depth, what factor (**1/nfac)?
    # nfac = 4
    # x_tr[:,0]   = x_tr[:,0]**(1/nfac) 
    # x_val[:,0]  = x_val[:,0]**(1/nfac) 
    # x_test[:,0] = x_test[:,0]**(1/nfac) 
    # if add_Tnoscat:
    #     xmin = np.array([0.0,0,0,0,0])
    #     xmax = np.array([18.5,1,1,1,1])
    # else:
    #     xmin = np.array([0.0,0,0,0])
    #     xmax = np.array([18.5,1,1,1])

    # log scaling instead 
    nfac = 1
    x_tr[:,0]   = np.log(x_tr[:,0]); 
    x_val[:,0]  = np.log(x_val[:,0])
    x_test[:,0] = np.log(x_test[:,0])
    if add_Tnoscat:
        xmin = np.array([-20.723267, 0, 0, 0, 0])
        # xmax = np.array([9.0,  1,  0.7,  0.99999514, 1.0])
        xmax = np.array([9.0,  1,  1.0,  1.0, 1.0])

    else:
        xmin = np.array([-20.723267, 0, 0, 0])
        xmax = np.array([11.695239,  1,  0.54999859,  0.99999514])
        
    # x_tr,xmin,xmax = preproc_minmax_inputs(x_tr, nfac_tau)
    x_tr    = preproc_minmax_inputs(x_tr,   nfac, (xmin,xmax))
    x_val   = preproc_minmax_inputs(x_val,  nfac, (xmin,xmax))
    x_test  = preproc_minmax_inputs(x_test, nfac, (xmin,xmax))
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
if scale_outputs:
    nfac = 2
    # nfac = 4
    
    # y_mean = np.array([0.04795693, 0.53649735, 0.06000423, 0.01418009]),
    #       dtype=np.float32)
    # y_sigma = np.array([0.3273185, 0.3273185, 0.3273185, 0.3273185],
    #       dtype=np.float32)
    
    y_mean = np.zeros(ny)
    y_sigma = np.zeros(ny)
    for i in range(ny):
        # y_mean[i] = (y_tr_raw[:,i]**(1/nfac)).mean()
        y_sigma[i] = (y_tr_raw[:,i]**(1/nfac)).std()
        # y_sigma[i] = (y_tr_raw[:,:]**(1/nfac)).std() 
    
    # No standard-scaling, just square root scaling
    y_mean  = np.repeat(0.0,ny)
    # y_sigma = np.repeat(1, ny)
    
    y_tr    = preproc_pow_gptnorm(y_tr_raw, nfac, y_mean, y_sigma)
    y_val   = preproc_pow_gptnorm(y_val_raw, nfac, y_mean, y_sigma)
    y_test  = preproc_pow_gptnorm(y_test_raw, nfac, y_mean, y_sigma)
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw
    


gc.collect()

          
# Inspect distributions of input and output variables?
if plot_distributions:
    # fig, ax = plt.subplots()
    # ax.hist(x_raw[:,0], bins=np.logspace(start=-6, stop=6, num=10))
    # ax.set_xscale('log'); ax.set_title("Optical depth (raw)")
    # ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6])
    xvars = ['tau scaled','ssa', 'g',    'mu']
    yvars = ['Rdif',      'Tdif','Rdir','Tdir']
    fig, ax = plt.subplots(2,2)
    fig2, ax2 = plt.subplots(2,2)
    i = 0
    for ix in range(2):
        for iy in range(2):
            ax[ix,iy].hist(x_tr[:,i])
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].set_title("{}".format(xvars[i]))
            
            ax2[ix,iy].hist(y_tr[:,i])
            ax2[ix,iy].set_yscale('log')
            # ax2[ix,iy].set_xlim([0.0,1.0])
            ax2[ix,iy].set_title("{}".format(yvars[i]))
            i = i + 1
    # Tdir values are low compared to Wiebkes master thesis??
    # x vals for higher Tdir: tau 1.5, ssa 1, g 0.993, mu 0.615

    for i in range(4):
        print("i={} : y_tr {} y_test {}".format(i,y_tr[:,i].std(),y_test[:,i].std()))


# Ready for training

# PYTORCH
if (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP#, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    lr          = 0.001
    batch_size  = 1024
    nneur       = 12
    mymodel = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.Softsign(), # first hidden layer
          nn.Linear(nneur, ny),
          nn.ReLU()
        )
#    mymodel = nn.Sequential(
#          nn.Linear(nx, nneur),
#          nn.Softsign(), # first hidden layer
#          nn.Linear(nneur, nneur),
#          nn.ReLU(), # second hidden layer
#          nn.Linear(nneur, ny)
#        )
    
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
    
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
            val_dataloaders=DataLoader(data_val,batch_size=batch_size))
    
    # Test model
    y_pred = mlp(x_test_torch)
    y_pred = y_pred.detach().numpy()
    if scale_outputs:
        y_pred      = preproc_pow_gptnorm_reverse(y_pred,nfac, y_mean,y_sigma)
        
    # # SAVE MODEL TO NEURAL-FORTRAN ASCII MODEL FILE
    # # Proved to be tricky..first convert to ONNX
    # import onnx
    # from onnx2keras import onnx_to_keras   
    # torch.onnx.export(mlp, x_test_torch, "tmp.onnx") 
    # onnx_model = onnx.load('tmp.onnx')
    # # Convert ONNX model to functional keras model
    # k_model_func = onnx_to_keras(onnx_model, ['input.1'])
    # # I couldn't figure out how to convert this to a Sequential keras model, so 
    # # we need to create one manually and then load the weights..
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # from ml_trainfuncs_keras import savemodel
    # k_model_seq = Sequential()
    # k_model_seq.add(Dense(nneur, input_dim=nx, activation='softsign'))
    # k_model_seq.add(Dense(ny, activation='relu'))
    # k_model_seq.set_weights(k_model_func.get_weights())
    # # Finally we can use save to a Neural-Fortran ascii model file
    # savemodel(fname, k_model_seq)

# TENSORFLOW-KERAS
elif (ml_library=='tf-keras'):
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    from tensorflow.keras import losses, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    
    # switch from GPU to CPU
    # from tensorflow.python.eager import context
    # _ = tf.Variable([1])
    # context._context = None
    # context._create_context()
    # # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # device_lib.list_local_devices()
    
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    
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
        
    # First hidden layer (input layer) activation
    activ0      = 'softsign'
    # activ0       = 'relu'
    
    # Activation in other hidden layers
    activ       =  activ0
    
    # Activation in last layer
    # activ_last   = 'linear'
    # activ_last = 'softsign'
    activ_last = 'relu'
    # activ_last = 'sigmoid'
    activ_last = 'hard_sigmoid'
    
    epochs      = 100000
    patience    = 15
    lossfunc    = losses.mean_squared_error
    lr          = 0.001
    # lr          = 0.0001 
    # batch_size  = 512
    batch_size  = 1024
    # neurons     = [16,16]
    neurons     = [8,8] # not quite fast enough, but accurate
    neurons     = [6,6]
    # neurons     = [12]
    # neurons     = [4,4] # nope
    # neurons = [8]

    optim = optimizers.Adam(lr=lr)
    
    # Create and compile model
    # model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
    #                          activ_last = activ_last, kernel_init='he_uniform')
    model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                             activ_last = activ_last, kernel_init='lecun_uniform')
    model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
    model.summary()
    
    # Create earlystopper
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    callbacks = [earlystopper]

    # Profiling
    # from datetime import datetime
    # log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)
    # callbacks = [callbacks[0], tensorboard_callback]
    
    # START TRAINING
    with tf.device(devstr):
        history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                            validation_data=(x_val,y_val), callbacks=callbacks)
    gc.collect()
    
    # PREDICT OUTPUTS FOR TEST DATA
    y_pred = model.predict(x_test);  
    if scale_outputs:
        y_pred = preproc_pow_gptnorm_reverse(y_pred,nfac, y_mean,y_sigma)
  
    # SAVE MODEL
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-12-logtau-sqrt-hardsig.h5"
    kerasfile = "/home/puk/soft/rte-rrtmgp-nn/neural/data/reftrans-6-6-logtau-sqrt-hardsig.h5"

    savemodel(kerasfile, model)
    

    # from keras.models import load_model
    # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # model = load_model(kerasfile,compile=False)
    
else:
    print("ml_library must be either 'pytorch' or 'tf-keras'")


# EVALUATE
for i in range(4):
    r = np.corrcoef(y_test_raw[:,i],y_pred[:,i])[0,1]
    print("R2 i={}: {} ; maxdiff {}".format(i,r**2,np.max(np.abs(y_test_raw[:,i]-y_pred[:,i]))))   
    if plot_eval:
        plot_hist2d(y_test_raw[:,i],y_pred[:,i],20,True) 
    
 # y_pred[:,i].mean(), y_pred[:,i].max(), y_pred[:,i].min()
