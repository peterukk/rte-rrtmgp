#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:39:06 2021

@author: peter
"""
import pickle

file_name = '/media/peter/samsung/data/reftrans_inout.pkl'
open_file = open(file_name, "rb")
data_list = pickle.load(open_file)
open_file.close()

print(data_list)

x_tr, y_tr, x_val, y_val, x_test, y_test, y_test_raw = data_list

nx = x_tr.shape[1]
ny = y_tr.shape[1]

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from ml_trainfuncs_keras import create_model_mlp, savemodel, mse_weights, \
  mae_weights2, mse_sineweight, mse_sigweight, mae_weights, mse_sineweight_nfac2, mse_sineweight_nfac2_2



import os
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
    


# First hidden layer (input layer) activation
activ0      = 'softsign'
# activ0      = 'relu'

# Activation in other hidden layers
activ       =  activ0

# Activation in last layer
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
batch_size  = 2048
batch_size  = 4096

# neurons     = [16,16]
neurons     = [8,8] # not quite fast enough, but accurate
# neurons     = [16]
retrain_mae = False
neurons     = [12,12]


# lossfunc = losses.mean_absolute_error
# valfunc     = 'val_mean_squared_error'
# mymetrics   = ['mean_squared_error']

lossfunc = mse_sineweight_nfac2_2

optim = optimizers.Adam(learning_rate=lr)

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


# START TRAINING
with tf.device(devstr):
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=callbacks)

    
# # PREDICT OUTPUTS FOR TEST DATA
# y_pred = model.predict(x_test);  
# if scale_outputs:
#     y_pred = preproc_pow_gptnorm_reverse(y_pred,nfac, y_mean,y_sigma)
  
# # ----- SAVE MODEL ------
# # kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-logtau-sqrt-mse-hardsig.h5"
# kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-msesine2.h5"

# # kerasfile = "/home/puk/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-logtau-sqrt-std.h5"
# savemodel(kerasfile, model)
# # -----------------------

# # ----- LOAD MODEL ------
# from tensorflow.keras.models import load_model
# kerasfile = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/reftrans-8-8-logtau-sqrt-mae.h5"
# model = load_model(kerasfile,compile=False)
# # model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
# # -----------------------


# # EVALUATE
# for i in range(4):
#     r = np.corrcoef(y_test_raw[:,i],y_pred[:,i])[0,1]
#     print("R2 {}: {:0.5f} ; maxdiff {:0.5f}, bias {:0.5f}".format(yvars[i], \
#       r**2,np.max(np.abs(y_test_raw[:,i]-y_pred[:,i])), np.mean(y_test_raw[:,i]-y_pred[:,i])))   
#     # if plot_eval:
#     #     plot_hist2d(y_test_raw[:,i],y_pred[:,i],20,True) 
#     #     plt.suptitle("{}".format(yvars[i]))
        
# plot_hist2d_reftrans(y_test_raw,y_pred,50,True) 