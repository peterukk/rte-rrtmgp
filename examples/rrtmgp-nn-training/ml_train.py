"""
Python code for developing neural networks to replace RRTMGP 
gas optics look up table

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from ml_load_save_preproc import save_model_netcdf, \
    load_rrtmgp, scale_outputs, \
    preproc_pow_standardization_reverse,\
    preproc_tau_to_crossection, preproc_minmax_inputs_rrtmgp
from ml_scaling_coefficients import * #  ymeans_sw_absorption, ysigma_sw_absorption, \
   # ymeans_sw_ray, ysigma_sw_ray, ymeans_lw_absorption, ysigma_lw_absorption

from ml_eval_funcs import plot_hist2d, plot_hist2d_T
from ml_trainfuncs_keras import create_model_mlp, savemodel, get_stdout


# ----------------------------------------------------------------------------
# ----------------- Provide data with inputs and outputs ---------------------
# ----------------------------------------------------------------------------

datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/peter/data/"


# Just one dataset
datadir = "/media/peter/samsung/data/ml_training/reduced-k/"
# fpath   = datadir+"ml_training_lw_g128_CKDMIP_MMM_big.nc"
fpath   = datadir+"ml_training_lw_g128_RFMIP-halton.nc"

fpath   = datadir+"ml_training_lw_g128_AMON_ssp245_ssp585_2054_2100.nc"

fpath_val = None
fpath_test = None

# ------------------------------------------------------
# --- Configure predictand, choice of scaling etc. -----
# ------------------------------------------------------


# Choose one of the following predictands (target output)
# 'lw_absorption', 'lw_planck_frac', 'sw_absorption', 'sw_rayleigh'
# predictand = 'sw_absorption'
# predictand = 'sw_rayleigh'

predictand = 'lw_absorption'
# predictand = 'lw_planck_frac'
# predictand = 'lw_both'

# Scaling method: currently fixed so that inputs are min-max scaled to (0..1) and 
# outputs are standardized to ~zero mean, ~unit variance
# However, power transformations are first used to reduce dynamical range 
# and make the distributions more Gaussian (can improve model convergence)
#
# Input preprocessing:
#   x(i) = log(x(i)) for input feature i = pressure
#   x(i) = x(i)**(1/4) for i = H2O and O3
#   x(i) = x(i) - max(i) / (max(x(i) - min(x(i))))
# 
# Output preprocessing:_
#   1. Normalize optical depths (g-point vectors) by layer number of molecules
#      y(ig,isample) = y_raw(ig,isample) / N (isample) 
#   2. y = y**(1/8)
#   3. ynorm = (y - ymean) / ystd, where ymeans are means for individual
#   g-points, but ystd is the standard deviation across all g-points 
#   (preserves relationships between outputs)
scaling_method = 'Ukkonen2020'

# scale_input = True
# scale_output = True

use_existing_input_scaling_coefficients = True 
# ^True is generally a safe choice, min max coefficients have been computed
# using a large dataset spanning both LGM (Last Glacial Maximum) and high 
# future emissions scenarios. However, check that your scaled inputs 
# fall somewhere in the 0-1 range

# Model training: use CPU or GPU?
use_gpu = False
num_cpu_threads = 12

retrain_mae = False

early_stop_on_rfmip_fluxes = True

# -----------------------------------------------------
# --------- Load data ---------------------------------
# -----------------------------------------------------
# Load data given three separate datasets for training - validation - testing
# Training data
x_tr_raw, y_tr_raw, col_dry_tr, input_names, kdist     = load_rrtmgp(fpath, predictand) 

    
nx = x_tr_raw.shape[1] #  temperature + pressure + gases
ny = y_tr_raw.shape[1] #  number of g-points

# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
if scaling_method != 'Ukkonen2020':
    print ("Only one type of pre-processing currently supported!")
else:
    # Input scaling - min-max
    if use_existing_input_scaling_coefficients:
        if xcoeffs_all == None:
            sys.exit("Input scaling coefficients (xcoeffs) missing!")
        (xmin_all,xmax_all) = xcoeffs_all
        # input_names loaded from file, describes inputs in order of x_tr_raw
        # input_names_all corresponds to xmin_all and xmax_all
        # input_names = ['tlay','play','n2o','co2']
        # Order of inputs may be different than in the existing coefficients:
        a = np.array(input_names_all)
        b = np.array(input_names)
        indices = np.where(b[:, None] == a[None, :])[1]
        xmin = xmin_all[indices]; xmax = xmax_all[indices]
        
        x_tr = preproc_minmax_inputs_rrtmgp(x_tr_raw, (xmin,xmax))
    else:
        x_tr,xmin,xmax  = preproc_minmax_inputs_rrtmgp(x_tr_raw)

    # Output scaling
    # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
    # log scaling. nfac = 8 for cross-sections, 2 for Planck fraction
    # After this, use standard-scaling
    if (predictand == 'lw_planck_frac'):
        nfac = 2
        ymean = None; ystd = None
        y_tr    = scale_outputs(y_tr_raw, None, nfac, ymean, ystd)
    elif (predictand == 'lw_both'):
        nfac = 4
        ymean = None; ystd = None
        ymean = np.zeros(ny); ystd = np.zeros(ny)
        nyy = int(ny/2)
        y_tr = y_tr_raw.copy()

        y_tr[:,0:nyy] = preproc_tau_to_crossection(y_tr[:,0:nyy], col_dry_tr)

        for i in range(ny):
            ymean[i] = np.mean(y_tr[:,i]**(1/nfac))
            ystd[i]  = np.std(y_tr[:,i]**(1/nfac))
        # ystd = np.repeat(np.std(y_tr**(1/nfac)),ny)
        
        # Scale data
        y_tr[:,0:nyy]   = scale_outputs(y_tr_raw[:,0:nyy], col_dry_tr, nfac, ymean[0:nyy], ystd[0:nyy])
        y_tr[:,nyy:]    = scale_outputs(y_tr_raw[:,nyy:], None, nfac, ymean[nyy:], ystd[nyy:])
        
    else:  # For scaling optical depths
        nfac = 8
        ymean = np.zeros(ny); ystd = np.zeros(ny)
        y_tr   = preproc_tau_to_crossection(y_tr_raw, col_dry_tr)

        for i in range(ny):
            ymean[i] = np.mean(y_tr[:,i]**(1/nfac))
            # ystd[i]  = np.std(y_tr[:,i]**(1/nfac))
        ystd = np.repeat(np.std(y_tr**(1/nfac)),ny)
                
        # Scale data
        y_tr    = scale_outputs(y_tr_raw, col_dry_tr, nfac, ymean, ystd)
    

# ---------------------------------------------------------------------------

# I/O: RRTMGP-NN models are saved as NetCDF files which contain metadata
# describing how to obtain the physical outputs, as well as the training data
x_scaling_str = "To get the required NN inputs, do the following: "\
        "x(i) = log(x(i)) for i=pressure; "\
        "x(i) = x(i)**(1/4) for i=H2O and O3; "\
        "x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i))"
if predictand == 'lw_planck_frac':
    y_scaling_str = "Model predicts the square root of Planck fraction."        
else:
    y_scaling_str = "Model predicts scaled cross-sections. Given the raw NN output y,"\
            " do the following to obtain optical depth: "\
            "y(igpt,j) = ystd(igpt)*y(igpt,j) + ymean(igpt); y(igpt,j) "\
            "= y(igpt,j)**8; y(igpt,j) = y(igpt,j) * layer_dry_air_molecules(j)"
        
data_str = "Extensive training data set comprising of reanalysis, climate model,"\
    " and idealized profiles, which has then been augmented using statistical"\
    " methods (Hypercube sampling). See https://doi.org/10.1029/2020MS002226"

if (predictand == 'sw_absorption'):
    model_str = "Shortwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'sw_rayleigh'):
    model_str = "Shortwave model predicting RAYLEIGH CROSS-SECTION"
elif (predictand == 'lw_absorption'):
    model_str = "Longwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'lw_planck_frac'):
    model_str = "Longwave model predicting PLANCK FRACTION"
else: 
    model_str = ""


        

# ---------------------------------------------------------------------------
# TENSORFLOW-KERAS TRAINING
# ------------------------------------------------------
# --- Model architecture -------------------------------
# ------------------------------------------------------
# Number of neurons in each hidden layer
# Activation functions after input layer and hidden layers respectively
activ = ['softsign', 'softsign','linear']
    
if predictand == 'lw_absorption':
    neurons     = [48,48}
else:
    neurons     = [16,16]

initializer = 'lecun_uniform'

# ------------------------------------------------------
# --- Loss function, batch size and learning rate ------
# ------------------------------------------------------
lossfunc    = losses.mean_squared_error
lr          = 0.001 
batch_size  = 1024
# batch_size  = 4096
# lr          = 0.01

# ------------------------------------------------------
# --- Early stopping : patience and what to monitor ----
# ------------------------------------------------------
patience    = 15
valfunc     = 'val_mean_absolute_error'
epochs      = 800  # set a high number with early stopping
# ------------------

# ------------------------------------------------------
# --- Custom metrics: would be great if we could monitor  
# --- flux errors, but that would require radiation code 
# ------------------------------------------------------
mymetrics   = ['mean_absolute_error']
# batch_size  = 3*batch_size
# lr          = 2 * lr

if use_gpu:
    devstr = '/gpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

else:
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
optim = optimizers.Adam(learning_rate=lr)

# Create and compile model
model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ=activ,
                         kernel_init=initializer)
model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
model.summary()

# Create earlystopper and possibly other callbacks
if early_stop_on_rfmip_fluxes:
    from ml_trainfuncs_keras import RunRadiationScheme

    fpath_save_tmp = '../../neural/data/tmp_model.nc'
    if predictand == 'lw_absorption':
        modelinput = '{} ../../neural/data/lw-g128-pfrac-tmp.nc'.format(fpath_save_tmp)
    elif predictand == 'lw_planck_frac':
        modelinput = '../../neural/data/lw-g128-abs-tmp.nc {}'.format(fpath_save_tmp)
    else:
        1

    def model_saver(fpath_save_tmp, model):
        save_model_netcdf(fpath_save_tmp, model, activ, input_names, kdist,
                               xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                               x_scaling_comment=x_scaling_str,
                               data_comment=data_str, model_comment=model_str)

    cmd = './rrtmgp_lw_eval_nn_rfmip 8 ../../rrtmgp/data/rrtmgp-data-lw-g128-210809.nc 1 1 ' + modelinput
    # out,err = get_stdout(cmd)
    # err_metrics_str = out[-23:-1].split('  ')
    # err_metrics = np.float32(err_metrics_str)
    patience = 20
    callbacks = [RunRadiationScheme(cmd, modelpath=fpath_save_tmp, 
                                        modelsaver=model_saver,
                                        patience=patience)]
    
    
else:
    callbacks = []
    
# ------------------------------------------------------
# --- Start training -----------------------------------
# ------------------------------------------------------
with tf.device(devstr):
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, 
                        shuffle=True,  verbose=1, callbacks=callbacks)     
    
        
        
# # ------------------------------------------------------
# # --- Evaluate on another dataset  ---------------------
# # ------------------------------------------------------
# def eval_valdata():
#     y_pred       = model.predict(x_val);  
#     y_pred       = preproc_pow_standardization_reverse(y_pred, nfac, ymean, ystd)
    
#     if predictand not in ['planck_frac', 'ssa_sw']:
#         y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))
        
#     plot_hist2d(y_val_raw,  y_pred,20,True)   # Optical depth 
#     plot_hist2d_T(y_val_raw,y_pred,20,True)   # Transmittance  
    
# eval_valdata()

# ------------------------------------------------------
# --- Save model?  -------------------------------------
# ------------------------------------------------------
model.summary()

fpath_keras = "../../neural/data/tau-lw-g128-tmp.h5"
model.save(fpath_keras)

fpath_netcdf = fpath_keras[:-3]+".nc"

print("Saving model from best epoch in both netCDF and HDF5 format to {}".format(fpath_netcdf))
save_model_netcdf(fpath_save_tmp, model, activ, input_names, kdist,
                       xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                       x_scaling_comment=x_scaling_str,
                       data_comment=data_str, model_comment=model_str)



