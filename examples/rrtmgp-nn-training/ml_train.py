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
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from ml_load_save_preproc import save_model_netcdf, \
    load_rrtmgp, preproc_pow_standardization_reverse,scale_gasopt
from ml_scaling_coefficients import  ymeans_sw_absorption, ysigma_sw_absorption, \
    ymeans_sw_ray, ysigma_sw_ray

from ml_eval_funcs import plot_hist2d, plot_hist2d_T
from ml_trainfuncs_keras import create_model_mlp, savemodel


# ----------------------------------------------------------------------------
# ----------------- Provide data with inputs and outputs ---------------------
# ----------------------------------------------------------------------------

datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/peter/data/"

# fpath       = datadir+"ml_data_g224_CAMS_2012-2016_noclouds.nc"
# fpath_val   = datadir+"ml_data_g224_CAMS_2017_noclouds.nc"
# fpath_test  = datadir+"ml_data_g224_CAMS_2018_noclouds.nc"

fpath       = datadir+"RRTMGP_data_g224_CAMS_2009-2018_sans_2014-2015_RND.nc"
fpath_val   = datadir+"RRTMGP_data_g224_CAMS_2014_RND.nc"
fpath_test  = datadir+"RRTMGP_data_g224_CAMS_2015_RND.nc"


# Just one dataset
# fpath_val = None
# fpath_test = None

# ------------------------------------------------------
# --- Configure predictand, choice of scaling etc. -----
# ------------------------------------------------------

# Do the inputs need pre-processing? Might have already been scaled (within the Fortran code)
scale_inputs = True

# Do the outputs need pre-processing?
scale_outputs = True

# Choose one of the following predictands (target output)
# 'lw_absorption', 'lw_planck_frac', 'sw_absorption', 'sw_rayleigh'
predictand = 'sw_absorption'
# predictand = 'sw_rayleigh'

# Scaling: currently fixed so that inputs are min-max scaled,
# (some) outputs are first power-scaled and then standardized...see below
use_existing_scaling_coefficients = True

# Model training: use CPU or GPU?
use_gpu = False
num_cpu_threads = 12

retrain_mae = False
# -----------------------------------------------------
# --------- Load data ---------------------------------
# -----------------------------------------------------
# Load data given three separate datasets for training - validation - testing
# Training data
x_tr_raw,y_tr_raw,col_dry_tr        = load_rrtmgp(fpath, predictand) 

# Validation and testing: separate datasets may already exist
if (fpath_val != None and fpath_test != None): 
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


# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
# Input scaling - min-max

# Load input names from netCDF file!
input_names_sw =  ['tlay','play','h2o','o3','co2','n2o', 'ch4']
input_names = input_names_sw

if use_existing_scaling_coefficients:
    xmin = np.array([1.60E2, 5.15E-3, 1.01E-2, 4.36E-3, 1.41E-4, 0.00E0, 2.55E-8], dtype=np.float32)
    xmax = np.array([ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3,
             5.8135214E-7, 3.6000001E-6], dtype=np.float32) 
    xcoeffs = (xmin,xmax)
    
    # Output scaling 
    nfac = 8 # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
    # log scaling. Useful when the output is a vector which has a wide range 
    # of magnitudes across the vector elements (g-points)
    # After this, use standard-scaling
    if (predictand == 'sw_absorption'):
        ymean  = ymeans_sw_absorption 
        ystd   = ysigma_sw_absorption
    elif (predictand == 'sw_rayleigh'):
        ymean  = ymeans_sw_ray #
        ystd   = ysigma_sw_ray
    else: 
        print("SPECIFY Y SCALING COEFFICIENTS")
else:
    print("code missing")

# Scale data, depending on choices 
x_tr,y_tr       = scale_gasopt(x_tr_raw, y_tr_raw, col_dry_tr, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)
# val
x_val,y_val     = scale_gasopt(x_val_raw, y_val_raw, col_dry_val, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)
# test
x_test,y_test   = scale_gasopt(x_test_raw, y_test_raw, col_dry_test, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)

x_tr[x_tr<0.0] = 0.0
x_val[x_val<0.0] = 0.0
x_test[x_test<0.0] = 0.0

nx = x_tr.shape[1]
ny = y_tr.shape[1] #  number of g-points
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TENSORFLOW-KERAS TRAINING
# ------------------------------------------------------
# --- Model architecture -------------------------------
# ------------------------------------------------------
# Number of neurons in each hidden layer
neurons     = [16,16]
# Activation functions after input layer and hidden layers respectively
activ = ['softsign', 'softsign','linear']
initializer = 'lecun_uniform'

# ------------------------------------------------------
# --- Loss function, batch size and learning rate ------
# ------------------------------------------------------
lossfunc    = losses.mean_squared_error
# lr          = 0.001 
# batch_size  = 1024
batch_size  = 4096
lr          = 0.01

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
optim = optimizers.Adam(lr=lr)

# Create and compile model
model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ=activ,
                         kernel_init=initializer)
model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
model.summary()

# Create earlystopper and possibly other callbacks
earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
callbacks = [earlystopper]

# ------------------------------------------------------
# --- Start training -----------------------------------
# ------------------------------------------------------
with tf.device(devstr):
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=callbacks)    
    
if retrain_mae:
    model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
    callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)]
    with tf.device(devstr):
        history2 = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                            validation_data=(x_val,y_val), callbacks=callbacks)
        
# ------------------------------------------------------
# --- Evaluate on validation data  ---------------------
# ------------------------------------------------------
def eval_valdata():
    y_pred       = model.predict(x_val);  
    y_pred       = preproc_pow_standardization_reverse(y_pred, nfac, ymean, ystd)
    
    if predictand not in ['planck_frac', 'ssa_sw']:
        y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))
        
    plot_hist2d(y_val_raw,  y_pred,20,True)   # Optical depth 
    plot_hist2d_T(y_val_raw,y_pred,20,True)   # Transmittance  
    
eval_valdata()

# ------------------------------------------------------
# --- Save model?  -------------------------------------
# ------------------------------------------------------
model.summary()

# fpath_keras = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16-CAMS-NEW-mae.h5"
# fpath_keras = "/home/peter/soft/rte-rrtmgp-nn/neural/data/tau-sw-abs-tmp2.h5"

# model.save(fpath_keras)

from keras.models import load_model
fpath_keras = "../../neural/data/BEST_tau-sw-ray-7-16-16_2.h5"
fpath_keras = '../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.h5'

fpath_netcdf = fpath_keras[:-3]+".nc"

model = load_model(fpath_keras,compile=False)

if (predictand == 'sw_absorption'):
    ymean  = ymeans_sw_absorption; ystd   = ysigma_sw_absorption
elif (predictand == 'sw_rayleigh'):
    ymean  = ymeans_sw_ray; ystd   = ysigma_sw_ray
else: 
    print("SPECIFY Y SCALING COEFFICIENTS")

x_scaling_str = "To get the required NN inputs, do the following: "\
        "x(i) = log(x(i)) for i=pressure; "\
        "x(i) = x(i)**(1/4) for i=H2O and O3; "\
        "x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i))"
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
else: 
    model_str = ""

emulating = 'rrtmgp-data-sw-g224-2018-12-04.nc'

save_model_netcdf(fpath_netcdf, model, activ, input_names, emulating,
                       xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                       x_scaling_comment=x_scaling_str,
                       data_comment=data_str, model_comment=model_str)
    
#savemodel(fpath_keras, model)

