#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE+RRTMGP (entire radiation scheme)

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

from ml_loaddata import load_inp_outp_radscheme, preproc_minmax_inputs, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from ml_eval_funcs import plot_hist2d
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
# fpath       = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2012-2016_clouds.nc"
fpath       = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2009-2016_clouds_noreftrans.nc"
fpath_val   = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2017_clouds_reftrans.nc"
fpath_test  = "/media/peter/samlinux/data/data_training/ml_data_g224_CAMS_2018_clouds_reftrans.nc"

# fpath       = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2011-2013_clouds.nc"
# fpath_val   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"
# fpath_test   = "/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/ml_data_g224_CAMS_2018_clouds.nc"

fpath_test = "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2011-2013_RFMIPstyle.nc"

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
x_tr_raw, y_tr_raw = load_inp_outp_radscheme(fpath, scale_p_h2o_o3 = scale_inputs)

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw, y_val_raw   = load_inp_outp_radscheme(fpath_val, scale_p_h2o_o3 = scale_inputs)
    x_test_raw,y_test_raw  = load_inp_outp_radscheme(fpath_test, scale_p_h2o_o3 = scale_inputs)
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

# Number of inputs and outputs    
nx = x_tr_raw.shape[1]
ny = y_tr_raw.shape[1]   



if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    
    fpath_xcoeffs = "../../../neural/data/nn_radscheme_xmin_xmax.txt"
    xcoeffs = np.loadtxt(fpath_xcoeffs, delimiter=',')
    xmax = xcoeffs[542:]
    # xmin = xcoeffs[0:542]

    xmin = np.repeat(0.0, 542)

    
    x_tr            = preproc_minmax_inputs(x_tr_raw, (xmin, xmax))
    # x_tr, xmin,xmax = preproc_minmax_inputs(x_tr_raw)
    x_val           = preproc_minmax_inputs(x_val_raw,  (xmin,xmax)) 
    x_test          = preproc_minmax_inputs(x_test_raw, (xmin,xmax)) 
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
    
if scale_outputs: 
    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = y_tr_raw[:,i].mean()
    #     # y_sigma[igpt] = y_raw[:,igpt].std()
    # # y_mean = np.repeat(y_raw.mean(),ny)
    # y_sigma = np.repeat(y_tr_raw.std(),ny)  # 467.72
    y_mean = np.array([374.46596068, 374.45956871, 374.45462027, 374.4511992 ,
       374.45047469, 374.45502195, 374.4680579 , 374.49239519,
       374.53055628, 374.58516238, 374.65892284, 374.75524611,
       374.8792801 , 375.03542025, 375.22643646, 375.45378099,
       375.71533287, 376.00285752, 376.30151249, 376.58998858,
       376.84997017, 377.06088253, 377.1959924 , 377.2298344 ,
       377.14753112, 376.9435507 , 376.61342392, 376.0964268 ,
       375.33952337, 374.26344007, 372.82036231, 370.98542669,
       368.80749149, 366.3915064 , 363.87406595, 361.26581523,
       358.59414259, 355.66949237, 352.05744001, 347.27688142,
       341.25040552, 334.76377694, 328.82791973, 323.08134132,
       316.79175838, 309.09519245, 299.90917436, 288.99921467,
       274.59731862, 257.36481124, 240.59263275, 226.58893258,
       216.13279391, 208.64963063, 203.82791957, 200.9401672 ,
       199.41826418, 198.6591551 , 198.25545666, 197.94484266,
       197.83066985, 897.8497014 , 897.56832421, 897.18702832,
       896.65574775, 895.92692424, 895.01315111, 893.947432  ,
       892.81558392, 891.67935113, 890.56589866, 889.47075025,
       888.3641525 , 887.21249947, 885.9978321 , 884.72128377,
       883.3744413 , 881.95796854, 880.47806976, 878.94735622,
       877.38503736, 875.785125  , 874.16277448, 872.56365052,
       871.02421366, 869.54206684, 868.07279715, 866.51043363,
       864.7201463 , 862.55637703, 859.83278134, 856.37222482,
       852.03649377, 846.8212828 , 840.8255009 , 834.20929904,
       826.98212955, 819.20462721, 810.68386777, 800.99420578,
       789.71007101, 776.9060625 , 763.57679203, 750.85167967,
       738.33882549, 725.31670556, 710.885572  , 695.07722868,
       677.67695876, 656.78869705, 633.36680778, 611.07829043,
       592.37422528, 578.04482565, 567.34107156, 559.94750542,
       555.05698858, 552.03360581, 550.17168289, 548.97796299,
       548.11269961, 547.68542119], dtype=np.float32)
    y_sigma = np.repeat(431.14175665, ny)

    nfac = 1
    y_tr    = preproc_pow_gptnorm(y_tr_raw, nfac, y_mean, y_sigma)
    y_val   = preproc_pow_gptnorm(y_val_raw, nfac, y_mean, y_sigma)
    y_test  = preproc_pow_gptnorm(y_test_raw, nfac, y_mean, y_sigma)
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw
    

gc.collect()
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
    nneur       = 256
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
        y_pred   = preproc_pow_gptnorm_reverse(y_pred, nfac, y_mean,y_sigma)
        
        plot_hist2d(y_test_raw,y_pred,20,True)      #  
    
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
    # activ_last   = 'relu'


    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    ninputs     = x_tr.shape[1]
    # lr          = 0.001
    lr          = 0.0001 
    # lr          = 0.0002 
    batch_size  = 256
    batch_size  = 512
    neurons     = [182, 182]
    # neurons     = [256,256] #0.994935
    neurons     = [128, 128]  #0.99804565
    neurons     = [64, 64]  # 0.9952047
    neurons     = [128]     # 0.9980413605
                    # 0.9996379952819034          
    if use_gpu:
        devstr = '/gpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    else:
        num_cpu_threads = 6
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
    
    optim = optimizers.Adam(learning_rate=lr)
    
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
    y_pred      = model.predict(x_test);  
    y_pred      = preproc_pow_gptnorm_reverse(y_pred, nfac, y_mean,y_sigma)
    
    cc = np.corrcoef(y_test_raw.flatten(), y_pred.flatten())
    diff = np.abs(y_test_raw-y_pred)
    rmse = np.sqrt(((y_pred - y_test_raw) ** 2).mean())
    
    print("r {} max diff {} RMSE {}".format(cc[0,1],np.max(diff), rmse))
    
    plot_hist2d(y_test_raw,y_pred,20,True)      # 
    # MAE 8

    
    # SAVE MODEL
    kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/radscheme-128_2.h5"

    savemodel(kerasfile, model)
    
    from tensorflow.keras.models import load_model
    kerasfile = "s/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/radscheme-128.h5"
    model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
    model = load_model(kerasfile,compile=False)
    
