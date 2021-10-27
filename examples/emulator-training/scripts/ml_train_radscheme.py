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

from ml_loaddata import load_radscheme, preproc_minmax_inputs, \
    preproc_standardization, preproc_standardization_reverse
from ml_eval_funcs import heatingrate_stats, plot_flux_and_hr_error, mae, plot_heatingrate_error
import matplotlib.pyplot as plt

def calc_heatingrates(y, p):
    fluxup = y[:,0:61]
    fluxdn = y[:,61:]
    F = fluxdn - fluxup
    # dF = np.gradient(F,axis=1)
    # dp = np.gradient(p,axis=1)
    dF = F[:,1:] - F[:,0:-1] 
    dp = p[:,1:] - p[:,0:-1] 
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day, fluxup, fluxdn

def calc_heatingrates_from_netflux(F, p):
    # dF = np.gradient(F,axis=1)
    # dp = np.gradient(p,axis=1)
    dF = F[:,1:] - F[:,0:-1] 
    dp = p[:,1:] - p[:,0:-1] 
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
# datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/puk/data/"
datadir     = "/home/peter/data/"

fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
# fpath2       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015_2.nc"

fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.nc"
# fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"

# ----------- config ----------------------------------------------------

scale_inputs    = True
scale_outputs   = True

# Which ML library to use: select either 'pytorch',
# or 'tf-keras' for Tensorflow with Keras frontend
# ml_library = 'pytorch'
ml_library = 'tf-keras'

# Model training: use GPU or CPU?
use_gpu = False
# use_gpu = False

# just predict heating rates directly, ignoring fluxes?
predict_hr = False
# OR!!!
# predict net fluxes + surface downwelling flux?
predict_netflux = False

# ----------------------------------------------------
# additional options: NOT relevant if predict_hr=true !

# -- Normalize outputs by inc flux? in this case, no other preproc. needed
norm_by_incflux = True

# -- Use custom loss function which minimizes heating rate error 
# -- as well as flux error? (Assumes norm_by_incflux=True) 
hre_loss = True

if predict_hr:
    norm_by_incflux = False
    hre_loss = False
    scale_outputs = True
    
if predict_netflux:
    scale_outputs = True
    norm_by_incflux = True
# ----------------------------------------------------

# ----------- config ----------------------------------------------------

# Load data
x_tr_raw, y_tr_raw, pres_tr = load_radscheme(fpath,  \
                        scale_p_h2o_o3 = scale_inputs, return_pressures=True)

# x_tr_raw2, y_tr_raw2, pres_tr2 = load_radscheme(fpath2,  \
#                         scale_p_h2o_o3 = scale_inputs, return_pressures=True)
# x_tr_raw = np.concatenate((x_tr_raw,x_tr_raw2),axis=0)
# y_tr_raw = np.concatenate((y_tr_raw,y_tr_raw2),axis=0)
    
if (fpath_val != None): # If val and test data exists
    x_val_raw, y_val_raw, pres_val      = load_radscheme(fpath_val,  \
                            scale_p_h2o_o3 = scale_inputs, return_pressures=True)
else: # if we only have one dataset, split manually
    from sklearn.model_selection import train_test_split
    validation_ratio = 0.15
    x_tr_raw, x_val_raw, y_val_raw, y_val_raw = \
        train_test_split(x_tr_raw, y_tr_raw, test_size=validation_ratio)

# Number of inputs    
nx = x_tr_raw.shape[1]

# if norm_by_incflux: 
#     nx = nx + 1
#     x_tr_raw = np.hstack((x_tr_raw,     y_tr_raw[:,61].reshape(-1,1)))
#     x_val_raw = np.hstack((x_val_raw,   y_val_raw[:,61].reshape(-1,1)))

# ---------- INPUT SCALING --------------
if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    
    fpath_xcoeffs = "../../../neural/data/nn_radscheme_xmin_xmax.txt"
    # fpath_xcoeffs = "../../../neural/data/_nn_radscheme_xmin_xmax.txt"

    xcoeffs = np.loadtxt(fpath_xcoeffs, delimiter=',')
    xmax = xcoeffs[0:nx]
    xmin = np.repeat(0.0, nx)

    x_tr            = preproc_minmax_inputs(x_tr_raw, (xmin, xmax))
    # x_tr, xmin,xmax = preproc_minmax_inputs(x_tr_raw)
    x_val           = preproc_minmax_inputs(x_val_raw,  (xmin,xmax)) 
    del x_tr_raw, x_val_raw
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw

# ---------- OUTPUT SCALING --------------

if predict_hr:
    y_tr, rsu_tr, rsd_tr    = calc_heatingrates(y_tr_raw, pres_tr)
    y_val, rsu_val, rsd_val = calc_heatingrates(y_val_raw, pres_val)
    # Add TOA and sfc fluxes
    y_tr = np.concatenate((y_tr,rsu_tr[:,0:1], rsu_tr[:,-2:-1], rsd_tr[:,0:1], rsd_tr[:,-2:-1]),axis=1)
    y_val = np.concatenate((y_val,rsu_val[:,0:1], rsu_val[:,-2:-1], rsd_val[:,0:1], rsd_val[:,-2:-1]),axis=1)
    # standard scaling with individual means but sigma computed across all outputs
    ny = y_tr.shape[1]   

    fpath_ycoeffs = "../../../neural/data/nn_radscheme_hr_std_scaling_coeffs.txt"
    ycoeffs = np.loadtxt(fpath_ycoeffs, delimiter=',')
    y_mean = np.float32(ycoeffs[0:ny])
    y_sigma = np.float32(ycoeffs[ny]); y_sigma = np.repeat(y_sigma,ny)
    
    # hr_tr, rsu_tr, rsd_tr    = calc_heatingrates(y_tr_raw, pres_tr)
    # hr_val, rsu_val, rsd_val = calc_heatingrates(y_val_raw, pres_val)
    # # Add TOA and sfc fluxes
    # y_tr = np.concatenate((hr_tr,  rsd_tr,  rsu_tr),axis=1)
    # y_val = np.concatenate((hr_val,rsd_val, rsu_val),axis=1)
    # # standard scaling with individual means but sigma computed across all outputs
    # ny = y_tr.shape[1]   

    # fpath_ycoeffs = "../../../neural/data/nn_radscheme_hrflux_std_scaling_coeffs.txt"
    # ycoeffs = np.loadtxt(fpath_ycoeffs, delimiter=',')
    # y_mean = np.float32(ycoeffs[0:ny])
    # y_sigma = np.float32(ycoeffs[ny]); y_sigma = np.repeat(y_sigma,ny)
    
    # y_sigma = y_tr.std()
    # y_mean = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = y_tr[:,i].mean()
    
    y_tr    = preproc_standardization(y_tr, y_mean, y_sigma)
    y_val   = preproc_standardization(y_val, y_mean, y_sigma)
    
elif predict_netflux:
    y_tr = y_tr_raw[:,61:] - y_tr_raw[:,0:61]
    y_val = y_val_raw[:,61:] - y_val_raw[:,0:61]
    for i in range(61):
        y_tr[:,i] = y_tr[:,i] / y_tr_raw[:,61]
        y_val[:,i] = y_val[:,i] / y_val_raw[:,61]
    ny = y_tr.shape[1]   

else:
    ny = y_tr_raw.shape[1]   

    if scale_outputs: 
        if norm_by_incflux:
            rsd0_tr = y_tr_raw[:,61]#.reshape(-1,1)
            rsd0_val = y_val_raw[:,61]#.reshape(-1,1)
            y_tr = np.zeros_like(y_tr_raw)
            y_val = np.zeros_like(y_val_raw)
            for i in range(ny):
                y_tr[:,i] = y_tr_raw[:,i] / rsd0_tr
                y_val[:,i] = y_val_raw[:,i] / rsd0_val

            # y_tr    = y_tr_raw / np.repeat(rsd0_tr,y_tr_raw.shape[1], axis=1)
            # y_val   = y_val_raw / np.repeat(rsd0_val,y_val_raw.shape[1], axis=1)
            
        else:

            fpath_ycoeffs = "../../../neural/data/nn_radscheme_hr_std_scaling_coeffs.txt"
            ycoeffs = np.loadtxt(fpath_ycoeffs, delimiter=',')
            y_mean = np.float32(ycoeffs[0:ny])
            y_sigma = np.float32(ycoeffs[ny]); y_sigma = np.repeat(y_sigma,ny)
            
            y_tr    = preproc_standardization(y_tr_raw, y_mean, y_sigma)
            y_val   = preproc_standardization(y_val_raw, y_mean, y_sigma)
            del y_tr_raw

    else:
        y_tr    = y_tr_raw    
        y_val   = y_val_raw
    

if hre_loss:
    # pres_tr_grad = np.gradient(pres_tr,axis=1)
    # pres_val_grad = np.gradient(pres_val,axis=1)
    pres_tr_grad    = pres_tr[:,1:] - pres_tr[:,0:-1] 
    pres_val_grad   = pres_val[:,1:] - pres_val[:,0:-1] 

gc.collect()
# Ready for training


# TENSORFLOW-KERAS TRAINING
if (ml_library=='tf-keras'):
    import tensorflow as tf
    from tensorflow.keras import losses, optimizers, layers, Input, Model
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    import tensorflow.keras.backend as K
    import time
    
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
    if norm_by_incflux: activ_last   = 'sigmoid'
    
    epochs      = 100000
    patience    = 28
    lossfunc    = losses.mean_squared_error
    ninputs     = x_tr.shape[1]
    lr          = 0.001
    # lr          = 0.0001 
    # lr          = 0.0002 
    # batch_size  = 512
    batch_size  = 1024
    
    optim = optimizers.Adam(learning_rate=lr)

    neurons     = [192, 192]
    neurons     = [128, 128]  #0.99804565
    neurons     = [128, 128,  128] # 0.9996379952819034    
    # neurons     = [64, 64,  64]            
    # neurons = [192]
    
    if use_gpu:
        devstr = '/gpu:0'
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
    
    
    if hre_loss:
        
        def my_gradient_tf(a):
            return a[:,1:] - a[:,0:-1]
        
        def calc_heatingrates_tf_dp(y, dp):
            #  flux_net =   flux_up   - flux_dn
            F = tf.subtract(y[:,0:61], y[:,61:])
            dF = my_gradient_tf(F)
            dFdp = tf.divide(dF, dp)
            coeff = -844.2071713147411#  -(24*3600) * (9.81/1004)  
            dTdt_day = tf.multiply(coeff, dFdp)
            return dTdt_day
        
        def CustomLoss(y_true, y_pred, incflux, dpres):
            err_flux = K.sqrt(K.mean(K.square(y_true - y_pred)))
            
            # need to reshape incflux from (batchsize,)  to (batchsize, 122)            
            fluxbig= tf.repeat(incflux, 122, axis=1)
            
            flux_true = tf.math.multiply(fluxbig, y_true)
            flux_pred = tf.math.multiply(fluxbig, y_pred)
            
            HR_true = calc_heatingrates_tf_dp(flux_true, dpres)
            HR_pred = calc_heatingrates_tf_dp(flux_pred, dpres)
            err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
            
            # alpha = 1
            # alpha = 0.002

            # alpha = 0.001
            alpha = 0.0005 # hybridloss.h5
            # alpha = 0.0003 # hybridloss2
            # alpha   = 0.0007
            alpha = 0.0008 #  hybridloss3
            err = (alpha) * err_hr + (1 - alpha)*err_flux
                
            return err
        
        def rmse_hr(y_true, y_pred, incflux, dpres):
            
            fluxbig= tf.repeat(incflux, 122, axis=1)
            
            flux_true = tf.math.multiply(fluxbig, y_true)
            flux_pred = tf.math.multiply(fluxbig, y_pred)
            
            HR_true = calc_heatingrates_tf_dp(flux_true, dpres)
            HR_pred = calc_heatingrates_tf_dp(flux_pred, dpres)

            return K.sqrt(K.mean(K.square(HR_true - HR_pred)))
    
        
        dense   = layers.Dense(neurons[0], activation=activ0)
        inp     = Input(shape=(nx,))
        x       = dense(inp)
        # more hidden layers
        for i in range(1,np.size(neurons)):
            x       = layers.Dense(neurons[i], activation=activ)(x)
        out     = layers.Dense(ny, activation=activ_last)(x)
        target  = Input((ny,))
        dpres   = Input((60,))
        incflux = Input((1,))
        model   = Model(inputs=[inp,target,incflux, dpres], outputs=out)
        model.add_loss(CustomLoss(target,out,incflux,dpres))
        model.add_metric(rmse_hr(target,out,incflux,dpres),'rmse_hr')
        
        model.compile(loss=None, optimizer=optim)
        
        callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, \
                        verbose=1, mode='min',restore_best_weights=True)]
            
        # with tf.device(devstr):
        
        history = model.fit(x=[x_tr, y_tr, rsd0_tr, pres_tr_grad], y=None,    \
            epochs= epochs, batch_size=batch_size, shuffle = True,      \
            validation_data=[x_val, y_val, rsd0_val, pres_val_grad], callbacks=callbacks)
        
        y_pred      = model.predict([x_val, y_val, rsd0_val, pres_val_grad]);
        
    else:
        
        # model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
        #                           activ_last = activ_last, kernel_init='he_uniform')
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
            
        start = time.time()
        y_pred      = model.predict(x_val);  
        end = time.time()
        print(end - start)       
        
       
    if scale_outputs:
        if norm_by_incflux:
            y_pred = y_pred * np.repeat(y_val_raw[:,61].reshape(-1,1),y_val_raw.shape[1], axis=1)
        else:
            y_pred      = preproc_standardization_reverse(y_pred, y_mean,y_sigma)


    # TEST

        
    if not (predict_hr or predict_netflux):  
        plot_flux_and_hr_error(y_val_raw, y_pred, pres_val)
    
    
    # SAVE MODEL
    # extract weights to save as simpler model without custom functions
    # layers 2,3,4,5,9 have weights
    all_weights = []
    for layer in model.layers:
      w = layer.weights
      try:      
        w[0]
        all_weights.append(layer.get_weights())
      except:
        pass # not all layers have weights !

    newmodel = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                          activ_last = activ_last, kernel_init='lecun_uniform')
    i = 0
    for layer in newmodel.layers:
      w = layer.weights
      try:      
        w[0]
        layer.set_weights(all_weights[i])
        i = i + 1
      except:
        pass # not all layers have weights !
    
    
    
    # kerasfile = "../../../neural/data/radscheme-128-128-128-hybridloss_new.h5"
    # savemodel(kerasfile, newmodel)
    
    # save as ONNX model for later runtime comparison against RNN model using ORT
    
    # First as TensorFlow savemodel file
    # fpath = 'saved_model/FNN-radscheme-128-128-128-hybridloss_new'
    # fpath_onnx = fpath+".onnx"
    # model.save(fpath, save_format='tf')
    # now convert to onnx
    # os.system("python -m tf2onnx.convert --saved-model {} --output {} --opset 13".format(fpath,fpath_onnx)) 

    # TEST ONNX INFERENCE
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    # import onnxruntime as ort
    # opts = ort.SessionOptions()
    # opts.intra_op_num_threads = 1
    # opts.inter_op_num_threads = 1
    # opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # sess = ort.InferenceSession(fpath_onnx, sess_options=opts)
    # sess = ort.InferenceSession(fpath_onnx, providers=["CUDAExecutionProvider"])

    # start = time.time()
    # y_pred = sess.run(["dense_11"], {"dense_8_input": x_val})[0]
    # end = time.time()
    # print(end - start)


    # Load existing model
    # from tensorflow.keras.models import load_model
    # kerasfile = "../../../neural/data/radscheme-128-128-128-fluxnorm-hybridloss.h5"
    # # model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
    # model = load_model(kerasfile,compile=False)

    
    
# PYTORCH TRAINING
elif (ml_library=='pytorch'):
    from torch import nn
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP#, MLP_cpu
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    lr          = 0.001
    batch_size  = 512
    nneur       = 128
    mymodel = nn.Sequential(
          nn.Linear(nx, nneur),
          nn.Softsign(), # first hidden layer
          nn.Linear(nneur, nneur),
          nn.Softsign(), # second hidden layer
          # nn.Linear(nneur, ny) # output layer
          nn.Sigmoid(nneur, ny) # output layer

        )
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)

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

    
    

# PLOT 
# np.save('tmp_noscale.npy',y_pred)
# np.save('tmp_stdscale.npy',y_pred)
# np.save('tmp_toascale.npy',y_pred)
# np.save('tmp_toascale_hrloss.npy',y_pred)
# np.save('tmp_hr.npy',y_pred)
# np.save('tmp_hr_ref.npy',y_val)



# from netCDF4 import Dataset

# rsd_pred_test = y_pred[:,61:]
# rsu_pred_test = y_pred[:,0:61]

# rootdir = "../fluxes/"
# fname_out = rootdir+'tmp.nc'
# # fname_out = rootdir+'CAMS_2015_rsud_RADSCHEME_RNN.nc'

# dat_out =  Dataset(fname_out,'a')
# var_rsu = dat_out.variables['rsu']
# var_rsd = dat_out.variables['rsd']

# nlay = 60
# nsite = dat_out.dimensions['site'].size
# ntime = dat_out.dimensions['time'].size
# var_rsu[:] = rsu_pred_test.reshape(ntime,nsite,nlay+1)
# var_rsd[:] = rsd_pred_test.reshape(ntime,nsite,nlay+1)

# dat_out.close()
