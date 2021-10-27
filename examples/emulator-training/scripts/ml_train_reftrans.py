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

from ml_loaddata import load_reftrans, preproc_minmax_inputs, \
    preproc_pow_standardization, preproc_pow_standardization_reverse, gen_synthetic_inp_outp_reftrans
from ml_eval_funcs import plot_hist2d, plot_hist2d_reftrans

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

datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
datadir     = "/home/peter/data/"

fpath_tr    = datadir+"REFTRANS_data_g224_CAMS_2009-2018_sans_2014-2015_RND.nc"
fpath_val   = datadir+"REFTRANS_data_g224_CAMS_2014_RND.nc"
fpath_test  = datadir+"/REFTRANS_data_g224_CAMS_2015_RND.nc"

# fpath_tr    = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2011-2013_clouds.nc"
# fpath_val   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"
# fpath_test  = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_RFMIP_noclouds.nc"

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
synthetic_data_supplement = False

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
plot_eval = True

# ----------- config ------------

# LOAD DATA
x_tr_raw,   y_tr_raw    = load_reftrans(fpath_tr, balance_samples)

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw,  y_val_raw   = load_reftrans(fpath_val, balance_samples)
    x_test_raw, y_test_raw  = load_reftrans(fpath_test)
    
    inds_val = np.isnan(y_val_raw[:,2])   
    y_val_raw = y_val_raw[~inds_val,:]; x_val_raw = x_val_raw[~inds_val,:]
    inds_test = np.isnan(y_test_raw[:,2])   
    y_test_raw = y_test_raw[~inds_test,:]; x_test_raw = x_test_raw[~inds_test,:]   
    
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
if synthetic_data_supplement:
    frac = 0.10
else:
    frac = 0.15



def extract_random_samples(x,y,frac):
    nrows       = x.shape[0]
    nkeep       = np.int(frac*nrows)
    
    inds_rare   = (y[:,0]>0.4) | (y[:,2]>0.4)
    inds_rare   = np.where(inds_rare)[0]
    
    inds        = np.random.default_rng().choice(nrows, nkeep, replace=False)
    inds_keep   = np.concatenate((inds,inds_rare))
    inds_keep.sort()

    x = x[inds_keep,:]
    y = y[inds_keep,:]
    return x,y

x_tr_raw,   y_tr_raw    = extract_random_samples(x_tr_raw, y_tr_raw, frac)
x_val_raw,  y_val_raw   = extract_random_samples(x_val_raw, y_val_raw, frac)
x_test_raw, y_test_raw  = extract_random_samples(x_test_raw, y_test_raw, 0.33*frac)

print( "{:e} training samples remain after trimming".format(x_tr_raw.shape[0]))


if synthetic_data_supplement:
    minmax_ssa  = (0.0, 1.0)
    minmax_g    = (0.0, 0.55)
    minmax_mu0  = (1e-3, 1.0)

    # The observed distribution has mostly small tau values
    minmax_tau  = (0.1, 20.0)
    nsamples    = np.int(1e6)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # Clear-sky conditions: g is zero
    minmax_tau  = (1e-09, 120000.00)
    nsamples = np.int(1e6)
    minmax_g    = None
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # larger g values
    minmax_ssa  = (0.35, 1.0)
    minmax_tau  = (1e-3, 100.0)
    minmax_g    = (0.4, 0.8)
    nsamples    = np.int(3e6)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_tr_raw = np.concatenate((x_tr_raw,x_raw2),axis=0)
    y_tr_raw = np.concatenate((y_tr_raw,y_raw2),axis=0)
    
    # EVEN LARGER g values
    minmax_ssa  = (0.9, 1.0)
    minmax_tau  = (0.05, 3.0)
    minmax_g    = (0.6, 0.9)
    minmax_mu0  = (0.9, 1.0)

    nsamples    = np.int(2e6)
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
    
# Add gammas as input if requested
use_gammas = False
if use_gammas:
    # xvars = ['tau scaled','ssa', 'g',    'mu']
    from ml_loaddata import reftrans_gammas
    gamma1_tr,gamma2_tr,gamma3_tr = reftrans_gammas(x_tr_raw[:,0], \
                x_tr_raw[:,1], x_tr_raw[:,2], x_tr_raw[:,3])

    gamma1_val,gamma2_val,gamma3_val = reftrans_gammas(x_val_raw[:,0], \
                x_val_raw[:,1], x_val_raw[:,2], x_val_raw[:,3])
    gamma1_test,gamma2_test,gamma3_test = reftrans_gammas(x_test_raw[:,0], \
                x_test_raw[:,1], x_test_raw[:,2], x_test_raw[:,3])
    # old inputs: tau, ssa, g,      mu0,    Tnoscat
    # new inputs: tau, ssa, gamma1, gamma2, gamma3, Tnoscat
   
    # add Tnoscat as input 6
    x_tr_raw = np.hstack((x_tr_raw,np.reshape(x_tr_raw[:,4],(x_tr_raw.shape[0],1))))
    x_val_raw = np.hstack((x_val_raw,np.reshape(x_val_raw[:,4],(x_val_raw.shape[0],1))))
    x_test_raw = np.hstack((x_test_raw,np.reshape(x_test_raw[:,4],(x_test_raw.shape[0],1))))
   
    # replace g with gamma1, mu0 with gamma2, old Tnoscat with gamma3
    x_tr_raw[:,2]   = gamma1_tr; x_tr_raw[:,3] = gamma2_tr; x_tr_raw[:,4] = gamma3_tr
    x_val_raw[:,2]  = gamma1_val; x_val_raw[:,3] = gamma2_val; x_val_raw[:,4] = gamma3_val
    x_test_raw[:,2] = gamma1_test; x_test_raw[:,3] = gamma2_test; x_test_raw[:,4] = gamma3_test

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
    nfac_tau = 4
    x_tr[:,0]   = x_tr[:,0]**(1/nfac_tau) 
    x_val[:,0]  = x_val[:,0]**(1/nfac_tau) 
    x_test[:,0] = x_test[:,0]**(1/nfac_tau) 
    if add_Tnoscat:
        xmin = np.array([0, 0, 0, 0, 0])
        # xmax = np.array([18.5,1,1,1,1])
        # xmax = np.array([13.05,1,1,1,1])
        xmax = np.array([13.05, 1, 1, 1, 1])

    else:
        xmin = np.array([0.0,0,0,0])
        xmax = np.array([13.05, 1, 1, 1])

    # log scaling instead 
    # nfac_tau = 1
    # x_tr[:,0]   = np.log(x_tr[:,0]); 
    # x_val[:,0]  = np.log(x_val[:,0])
    # x_test[:,0] = np.log(x_test[:,0])
    
    # if use_gammas:
    #     xmin = np.array([-20.723267,  0.,  0.07535875,  0.0,   -0.174691384,  0.0])
    #     xmax = np.array([11.91744,    1. , 2. ,   0.75,  0.5 ,       1.  ])
    # else:
    #     if add_Tnoscat:
    #         xmin = np.array([-20.723267, 0, 0, 0, 0])
    #         # xmax = np.array([9.0,  1,  0.7,  0.99999514, 1.0])
    #         xmax = np.array([9.0,  1,  1.0,  1.0, 1.0])
   
    #     else:
    #         xmin = np.array([-20.723267, 0, 0, 0])
    #         xmax = np.array([11.695239,  1,  0.54999859,  0.99999514])
            
    # if add_Tnoscat:
    #     xmin = np.array([-20.723267, 0, 0, 0, 0])
    #     # xmax = np.array([9.0,  1,  0.7,  0.99999514, 1.0])
    #     xmax = np.array([9.0,  1,  1.0,  1.0, 1.0])

    # else:
    #     xmin = np.array([-20.723267, 0, 0, 0])
    #     xmax = np.array([11.695239,  1,  0.54999859,  0.99999514])
        
    # x_tr,xmin,xmax = preproc_minmax_inputs(x_tr, nfac_tau)
    x_tr    = preproc_minmax_inputs(x_tr, (xmin,xmax))
    x_val   = preproc_minmax_inputs(x_val, (xmin,xmax))
    x_test  = preproc_minmax_inputs(x_test, (xmin,xmax))
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
    
if scale_outputs:
    nfac = 2
    # nfac = 4
    # nfac = 1
    
    # y_sigma = np.array([0.19369066, 0.43910795, 0.20916097, 0.22079377],
    #       dtype=np.float32)
    # y_sigma = np.array([0.10764305, 0.39538148, 0.13180389, 0.15161364],
    #       dtype=np.float32)
    
    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = (y_tr_raw[:,i]**(1/nfac)).mean()
    #     # y_sigma[i] = (y_tr_raw[:,i]**(1/nfac)).std()
    #     y_sigma[i] = (y_tr_raw**(1/nfac)).std()

    # No standard-scaling, just square root scaling
    y_mean  = np.repeat(0.0,ny)
    y_sigma = np.repeat(1, ny)
    
    # nfac2
    # y_sigma = np.array([0.15692602, 0.42003798, 0.17412843, 0.18447757], 
    #                     dtype=np.float32)
    # y_mean =  np.array([0.11233056, 0.63645709, 0.12254605, 0.11041685], 
    #                     dtype=np.float32)
    
    # nfac2, no mean
    # y_sigma = np.array([0.15692602, 0.42003798, 0.17412843, 0.18447757], 
    #                     dtype=np.float32)
    # y_mean = np.array([0., 0., 0., 0.], dtype=np.float32)
    
    # nfac2, single sigma
    # y_sigma = np.array([0.34254798, 0.34254798, 0.34254798, 0.34254798], 
    #                     dtype=np.float32)
    # y_mean =  np.array([0.11233056, 0.63645709, 0.12254605, 0.11041685], 
    #                     dtype=np.float32)
    
    #nfac1 
    # y_sigma = np.array([0.08930878, 0.43197197, 0.10786474, 0.12255082], 
    #                     dtype=np.float32)
    # y_mean =  np.array([0.03724393, 0.58150858, 0.04533824, 0.04622386],
    #                     dtype=np.float32)
    
    y_tr    = preproc_pow_standardization(y_tr_raw, nfac, y_mean, y_sigma)
    y_val   = preproc_pow_standardization(y_val_raw, nfac, y_mean, y_sigma)
    y_test  = preproc_pow_standardization(y_test_raw, nfac, y_mean, y_sigma)
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
            ax[ix,iy].hist(x_tr[:,i]**(1/2.0))
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
        print("{} : std y_tr {} y_test {}".format(yvars[i],y_tr[:,i].std(),y_test[:,i].std()))
        print("{} : mean y_tr {} y_test {}".format(yvars[i],y_tr[:,i].mean(),y_test[:,i].mean()))

# Ready for training

# PYTORCH
if (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    lr          = 0.001
    batch_size  = 1024
    # nneur       = 12
    # mymodel = nn.Sequential(
    #       nn.Linear(nx, nneur),
    #       nn.Softsign(), # first hidden layer
    #       nn.Linear(nneur, ny),
    #       nn.ReLU()
    #     )
    nneur = 8
    mymodel = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.Softsign(), # first hidden layer
          nn.Linear(nneur, nneur),
          nn.ReLU(), # second hidden layer
          nn.Linear(nneur, ny)
        )
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)
    
    x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
    data_test    = TensorDataset(x_test_torch,y_test_torch)
    
    mlp = MLP(nx=nx,ny=ny,learning_rate=lr,SequentialModel=mymodel)
    mlp = MLP_cpu(nx=nx,ny=ny,learning_rate=lr,SequentialModel=mymodel)

    mc = pl.callbacks.ModelCheckpoint(monitor='val_loss',every_n_epochs=2)
    
    if use_gpu:
        trainer = pl.Trainer(gpus=0, deterministic=True)
    else:
        num_cpu_threads = 6
        trainer = pl.Trainer(accelerator="ddp_cpu", callbacks=[mc], deterministic=True,
                num_processes=  num_cpu_threads)
                #plugins=pl.plugins.DDPPlugin(find_unused_parameters=False))
    
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
            val_dataloaders=DataLoader(data_val,batch_size=batch_size))
    
    # Test model
    y_pred = mlp(x_test_torch)
    y_pred = y_pred.detach().numpy()
    if scale_outputs:
        y_pred      = preproc_pow_standardization_reverse(y_pred,nfac, y_mean,y_sigma)
        
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
    from ml_trainfuncs_keras import create_model_mlp, savemodel, mse_weights, \
      mae_weights2, mse_sineweight, mse_sigweight, mae_weights, \
          mse_sineweight_nfac2, mse_sineweight_nfac2_2, mse_sineweight_nfac2_3
    import tensorflow.keras.backend as K

    # Set up training hardware
    if use_gpu:
        devstr = '/gpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    else:
        num_cpu_threads = 12
        devstr = '/cpu:0'
        # # Maximum number of threads to use for OpenMP parallel regions.
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
        
    # switch from GPU to CPU
    # from tensorflow.python.eager import context
    # _ = tf.Variable([1])
    # context._context = None
    # context._create_context()
    # # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # device_lib.list_local_devices()


    # First hidden layer (input layer) activation
    activ0      = 'softsign'
    # activ0      = 'relu'

    # Activation in other hidden layers
    activ       =  activ0
    
    # Activation in last layer
    # activ_last   = 'linear'
    # activ_last = 'softsign'
    # activ_last = 'relu'
    # activ_last = 'sigmoid'
    activ_last = 'hard_sigmoid'
    
    epochs      = 100000
    patience    = 15
    
    lossfunc    = losses.mean_squared_error
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'

    lr          = 0.001
    # lr          = 0.0001 
    # batch_size  = 512
    batch_size  = 1024
    neurons     = [8,8] # not quite fast enough, but accurate
    # neurons     = [16]
    neurons     = [12,12]
    # neurons     = [4,4] # nope
    # neurons = [8]
    retrain_mae = False
    
    # batch_size  = 4096
    batch_size = 1024
    batch_size = 2048
    # lr          = 0.01
    
    optim = optimizers.Adam(lr=lr)


    # lossfunc = losses.mean_absolute_error
    # valfunc     = 'val_mean_squared_error'
    # mymetrics   = ['mean_squared_error']

    # lossfunc = mse_sineweight
    # valfunc     = 'val_loss'

    # lossfunc = mse_weights
    # lossfunc = mae_weights # pretty ok
    # lossfunc = mae_weights2
    # lossfunc = mae_sine_and_y_weight
    # lossfunc = mse_sineweight_nfac2_2
    
    # def mse_sineweight_nfac2(y_true, y_pred):
    #     weights = 2.0 * (K.sin(1.0 * K.square(y_true) )) - 0.4
    #     wg = np.array([2.5, 1.0, 2.5, 2.0], dtype=np.float32)
    #     # wg = np.reshape(wg,(1,4)).repeat(batch_size,axis=0)
    #     y_true = y_true*wg
    #     y_pred = y_pred*wg
    #     return K.mean(K.square(weights*(y_true - y_pred)),axis=-1)

    def mse_sineweight_nfac2(y_true, y_pred):
        weights = 1.5 * (K.sin(1.1 * K.square(y_true) )) 
        wg = np.array([2.5, 1.0, 2.5, 2.0], dtype=np.float32)
        y_true = y_true*wg
        y_pred = y_pred*wg
        return K.mean(K.square(weights*(y_true - y_pred)),axis=-1)

    lossfunc =  mse_sineweight_nfac2
    
    def mae_weights(y_true,y_pred):
        wg = np.array([2.0, 1.0, 3.0, 2.0], dtype=np.float32)
        # wg = np.array([2.5, 1.0, 2.0, 2.5], dtype=np.float32)
        y_true = y_true*wg
        y_pred = y_pred*wg
        return K.mean(K.abs(y_true - y_pred),axis=0)
    
    lossfunc = losses.mean_absolute_error
    lossfunc = mae_weights
    valfunc     = 'val_loss'
    patience    = 21
    
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
    
    # (optional) recompile with MAE and continue training
    if retrain_mae:
        # loss_new = mae_weights
        # model.compile(loss=loss_new, optimizer=optim,metrics=['mean_squared_error'])
        model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
        callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)]
        with tf.device(devstr):
            history2 = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                                validation_data=(x_val,y_val), callbacks=callbacks)
        
    # PREDICT OUTPUTS FOR TEST DATA
    y_pred = model.predict(x_val);  
    if scale_outputs:
        y_pred = preproc_pow_standardization_reverse(y_pred,nfac, y_mean,y_sigma)
  
    # ----- SAVE MODEL ------
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-logtau-sqrt-mse-hardsig.h5"
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-12-12-mae2-NEW.h5"

    # kerasfile = "/home/puk/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-logtau-sqrt-std.h5"
    # savemodel(kerasfile, model)
    # -----------------------
    
    # ----- LOAD MODEL ------
    # from tensorflow.keras.models import load_model
    # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-12-12-sinemse2-mae-NEW.h5"
    # model = load_model(kerasfile,compile=True)
    # model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
    # -----------------------

else:
    print("ml_library must be either 'pytorch' or 'tf-keras'")


# EVALUATE
yvars = ['Rdif',      'Tdif','Rdir','Tdir']
for i in range(4):
    r = np.corrcoef(y_val_raw[:,i],y_pred[:,i])[0,1]
    print("R2 {}: {:0.5f} ; maxdiff {:0.5f}, bias {:0.5f}".format(yvars[i], \
      r**2,np.max(np.abs(y_val_raw[:,i]-y_pred[:,i])), np.mean(y_val_raw[:,i]-y_pred[:,i])))   
    # if plot_eval:
    #     plot_hist2d(y_test_raw[:,i],y_pred[:,i],20,True) 
    #     plt.suptitle("{}".format(yvars[i]))
        
plot_hist2d_reftrans(y_val_raw,y_pred,50,True) 

 # y_pred[:,i].mean(), y_pred[:,i].max(), y_pred[:,i].min()
