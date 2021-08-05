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
import os
import gc
import numpy as np

from ml_loaddata import load_inp_outp_reftrans, preproc_minmax_inputs_reftrans, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse, gen_synthetic_inp_outp_reftrans
from ml_eval_funcs import plot_hist2d

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, REFTRANS EMULATION  ------------------------
# ----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# ----------- config

scale_inputs = True
scale_outputs = True

# OPTIONAL
# CAMS data: balance samples so that half of them correspond to cloudy layers?
# This is done by extracting all cloudy layers, and then an equal amount of 
# random non-cloudy layers
balance_samples = True

# Generate data synthetically by doing hypercube sampling of inputs and 
# generating corresponding outputs on the fly? For REFTRANS computations this 
# is very doable because there's only 4 inputs; additionally the 
# reftrans routine can easily be coded in Python
synthetic_data = False
synthetic_data_supplement = True

# Use independent clear-sky RFMIP profiles for testing? Doesn't reflect
# the training data which includes cloudy samples, but may be interesting 
test_on_RFMIP = True

# Add no-scattering transmittance as a NN input?
add_Tnoscat = True


#  ----------------- File paths -----------------
fpath = "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc"  
fpath_rfmip = "/media/peter/samlinux/data/data_training/ml_data_reftrans_RFMIP.nc"
# dat_file = "ml_data_g224_CAMS_2018_clouds.nc"
# dat_dir  = '/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/'
# fpath = dat_dir + dat_file


# x_raw[:,i].min(), x_raw[:,i].max()
# i0 (4.1395123e-09, 29715.678)
# i1 (4.095197e-12, 1.0)
# i2 (0.0, 0.48579213)
# i3  (0.00012061903, 0.9999998)

# if synthetic_data:
#     nsamples    = np.int(3e6)
#     x_raw, y_raw = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
#                                     minmax_mu0)
#     minmax_tau  = (1.0, 10.0)
#     x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
#                                     minmax_mu0)
#     x_raw = np.concatenate((x_raw,x_raw2),axis=0)
#     y_raw = np.concatenate((y_raw,y_raw2),axis=0)
#     x_raw_test,y_raw_test =  load_inp_outp_reftrans(fpath)
#     # this "real" dataset is large, lets pick random 5% for testing
#     frac = 0.05 
#     nrows = x_raw_test.shape[0]
#     inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
#     x_raw_test = x_raw_test[inds_rand,:]; y_raw_test = y_raw_test[inds_rand,:]
# else:
    
x_raw, y_raw  = load_inp_outp_reftrans(fpath, balance_samples)

# this "real" data is large, lets pick random samples
if balance_samples:
    frac = 0.06
else:
    frac = 0.02
nrows = x_raw.shape[0]
inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
x_raw = x_raw[inds_rand,:]; y_raw = y_raw[inds_rand,:]


if test_on_RFMIP:
    x_raw_test,y_raw_test  = load_inp_outp_reftrans(fpath_rfmip)
    test_frac = 0.08
    nrows = x_raw_test.shape[0]
    inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
    x_raw_test = x_raw_test[inds_rand,:]; y_raw_test = y_raw_test[inds_rand,:]
    if add_Tnoscat:
        mu0_inv = (1/x_raw_test[:,3])
        tnoscat = np.exp(-x_raw_test[:,0]*mu0_inv)
        x_raw_test = np.hstack((x_raw_test,np.reshape(tnoscat,(inds_rand.size,1))))
else:
    # random 20% of original data for testing 
    test_frac = 0.20
    x_raw, x_raw_test, y_raw, y_raw_test = train_test_split(x_raw, y_raw, test_size=test_frac)

if synthetic_data_supplement:
    minmax_ssa  = (0.0, 1.0)
    minmax_g    = (0.0, 0.55)
    minmax_mu0  = (0.0, 1.0)

    # The observed distribution has mostly small tau values
    minmax_tau  = (0.1, 20.0)
    nsamples    = np.int(5e5)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                minmax_mu0)
    x_raw = np.concatenate((x_raw,x_raw2),axis=0)
    y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    
    # Clear-sky conditions: g is zero
    minmax_tau  = (1e-09, 120000.00)
    nsamples = np.int(5e5)
    minmax_g    = None
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_raw = np.concatenate((x_raw,x_raw2),axis=0)
    y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    minmax_g    = (0.0, 0.55)
    
    # larger g values
    minmax_ssa  = (0.35, 1.0)
    minmax_tau  = (1e-3, 100.0)
    minmax_g    = (0.4, 0.8)
    nsamples    = np.int(1e6)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_raw = np.concatenate((x_raw,x_raw2),axis=0)
    y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    
    # EVEN LARGER g values
    minmax_ssa  = (0.9, 1.0)
    minmax_tau  = (0.05, 3.0)
    minmax_g    = (0.6, 0.9)
    minmax_mu0  = (0.9, 1.0)

    nsamples    = np.int(5e5)
    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_raw = np.concatenate((x_raw,x_raw2),axis=0)
    y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    
    y_raw[y_raw<0.0] = 0.0


if add_Tnoscat:
    mu0_inv = (1/x_raw[:,3])
    tnoscat = np.exp(-x_raw[:,0]*mu0_inv)
    x_raw = np.hstack((x_raw,np.reshape(tnoscat,(x_raw.shape[0],1))))

if scale_inputs:
    # Square-root scaling of optical depth, what factor (**1/nfac)?
    # nfac_tau = 4
    # xmin = np.array([0.005, 0, 0, 0]) #n fac 4
    # xmax = np.array([18.5,  1,  0.54999859,  1])# nfac 4
    # xmax = np.array([4.0,  1,  0.54999859,  0.99999514]) # nfac 8
    x = np.copy(x_raw)
    x_test = np.copy(x_raw_test)
    
    # log scaling instead 
    nfac_tau = 1    
    x[:,0] = np.log(x[:,0]);  x_test[:,0] = np.log(x_test[:,0])
    if add_Tnoscat:
        xmin = np.array([-20.723267, 0, 0, 0, 0])
        # xmax = np.array([11.695239,  1,  0.54999859,  0.99999514, 1.0])
        xmax = np.array([9.0,  1,  0.7,  0.99999514, 1.0])

    else:
        xmin = np.array([-20.723267, 0, 0, 0])
        xmax = np.array([11.695239,  1,  0.54999859,  0.99999514])
        
    # x,xmin,xmax = preproc_minmax_inputs_reftrans(x_raw,nfac_tau)
    
    x       = preproc_minmax_inputs_reftrans(x, nfac_tau, (xmin,xmax))
    x_test  = preproc_minmax_inputs_reftrans(x_test, nfac_tau, (xmin,xmax))
else:
    x = x_raw
    x_test = x_raw_test
    
if scale_outputs:
    ny = y_raw.shape[1]    
    # original
    # nfac = 1
    nfac = 2
    # nfac = 4
    # nfac = 8
    # nfac = 16

    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = (y_raw[:,i]**(1/nfac)).mean()
    #     # y_sigma[i] = (y_raw[:,i]**(1/nfac)).std()
    #     y_sigma[i] = (y_raw[:,:]**(1/nfac)).std() 
    
    y_mean  = np.repeat(0.0,ny)
    y_sigma = np.repeat(1, ny)
    
    y       = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
    y_test  = preproc_pow_gptnorm(y_raw_test, nfac, y_mean, y_sigma)
else:
    y       = y_raw
    y_test  = y_raw_test
    
# y_mean = np.array([0.04795693, 0.53649735, 0.06000423, 0.01418009]),
#       dtype=np.float32)

# y_sigma = np.array([0.3273185, 0.3273185, 0.3273185, 0.3273185],
#       dtype=np.float32)


gc.collect()
(ns,ny) = y.shape
nx = x.shape[1]

# Validation data as a subset of the training data
val_ratio = 0.25
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.25)

          
# eval


# for i in range(4):
#     print("tr {} test {}".format(y_tr[:,i].std(),y_test[:,i].std()))
# for i in range(4):
#     print("RAW std {} tr {} test {}".format(i,y_raw[:,i].std(),y_raw_test[:,i].std()))
#     # print("RAW max {} tr {} test {}".format(i,y_raw[:,i].max(),y_raw_test[:,i].max()))

plot_distributions = False
if plot_distributions:
    # fig, ax = plt.subplots()
    # ax.hist(x_raw[:,0], bins=np.logspace(start=-6, stop=6, num=10))
    # ax.set_xscale('log'); ax.set_title("Tau")
    # ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6])
    
    xvars = ['tau scaled','ssa','g','mu']
    yvars = ['Rdif','Tdif','Rdir','Tdir']
    fig, ax = plt.subplots(2,2)
    fig2, ax2 = plt.subplots(2,2)
    i = 0
    for ix in range(2):
        for iy in range(2):

            ax[ix,iy].hist(x[:,i])
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].set_title("{}".format(xvars[i]))
            
            ax2[ix,iy].hist(y_raw[:,i])
            ax2[ix,iy].set_yscale('log')
            ax2[ix,iy].set_xlim([0.0,1.0])
            ax2[ix,iy].set_title("{}".format(yvars[i]))
            i = i + 1
    # Tdir values are low compared to Wiebkes master thesis??
    # x vals for higher Tdir: tau 1.5, ssa 1, g 0.993, mu 0.615
    
    # from fastkde.fastKDE import pdf
    # def get_pdf(data):
    #     y, x = pdf(data)
    #     return x, y
    
    # fig, ax = plt.subplots(2,2)
    # fig2, ax2 = plt.subplots(2,2)
    # i = 0
    # for ix in range(2):
    #     for iy in range(2):
    #         xp,yp = get_pdf(x_tr[:,i])
    #         ax[ix,iy].plot(xp,yp,label='Training')
    #         if (i != 2): 
    #             xp2,yp2 = get_pdf(x_test[:,i])
    #             ax[ix,iy].plot(xp2,yp2,label='Testing')
    #         ax[ix,iy].legend()
    #         ax[ix,iy].set_title("KDE for X={}".format(i))
    #         #outp
    #         xp,yp = get_pdf(y_tr[:,i])
    #         ax2[ix,iy].plot(xp,yp,label='Training')
    #         xp2,yp2 = get_pdf(y_test[:,i])
    #         ax2[ix,iy].plot(xp2,yp2,label='Testing')
    #         ax2[ix,iy].legend()
    #         ax2[ix,iy].set_xlim([0.0,1.0])
    #         ax2[ix,iy].set_title("KDE for Y={}".format(i))
    #         i = i + 1
    # plt.show()

# Ready for training


train_pytorch = False

if train_pytorch:
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP

    batch_size = 256
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)
    
    x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
    data_test    = TensorDataset(x_test_torch,y_test_torch)
    
    mlp = MLP(nx=nx,ny=ny)
    #trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
    
    trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=10)
    #trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=5,num_processes=3)
    
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
                val_dataloaders=DataLoader(data_val,batch_size=batch_size))



train_keras = True

if train_keras:
    
    from keras import losses, optimizers
    from keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel

    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    
    activ0      = 'softsign'

    # activ0      = 'sigmoid'
    # activ0       = 'relu'
    activ       =  activ0
    
    # activ_last = 'softsign'
    # activ_last = 'relu'
    activ_last = 'sigmoid'
    # activ_last   = 'linear'
    
    epochs      = 100000
    patience    = 15
    lossfunc    = losses.mean_squared_error
    
    lr          = 0.001
    # lr          = 0.002
    # lr          = 0.0001 
    # lr          = 0.0002 
    # batch_size  = 512
    batch_size  = 1024
    # neurons     = [16,16]
    # neurons     = [8]
    # neurons     = [8,8]
    # neurons     = [16]
    # neurons     = [6,6]
    neurons = [12]
    # neurons = [4,4]
    
    optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    # optim = optimizers.Adam(lr=lr)
    
    # Create model
    model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                             activ_last = activ_last, kernel_init='he_uniform')
    # Compile model
    model.compile(loss=lossfunc, optimizer=optim,
                  metrics=mymetrics,  context= ["gpu(0)"])
    model.summary()
    
    # Create earlystopper
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    
    # START TRAINING
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=[earlystopper])
    gc.collect()
    
    # TEST

    y_test_pred      = model.predict(x_test);  
    
    if scale_outputs:
        y_test_pred      = preproc_pow_gptnorm_reverse(y_test_pred,nfac, y_mean,y_sigma)
        
    # for i in range(4):
    #     plot_hist2d(y_raw_test[:,i],y_test_pred[:,i],20,True)      # [6] 0.9988


    for i in range(4):
        r = np.corrcoef(y_raw_test[:,i],y_test_pred[:,i])[0,1]
        print("R2 i={}: {}".format(i,r**2))    
        
    #0 0.36045932
    #1 0.9993
    #2 0.47
    #3 0.495156
    
    # y_test_pred[:,i].mean(), y_test_pred[:,i].max(), y_test_pred[:,i].min()
    
    # SAVE MODEL
    
    kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-12-logtau-sqrt.h5"
    savemodel(kerasfile, model)

    # from keras.models import load_model
    # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # model = load_model(kerasfile,compile=False)
    


