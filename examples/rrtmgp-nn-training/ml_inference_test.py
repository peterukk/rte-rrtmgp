#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:10:42 2022

@author: peter
"""
import os
import gc
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from ml_load_save_preproc import save_model_netcdf, \
    load_rrtmgp, preproc_pow_standardization_reverse,scale_gasopt
from ml_scaling_coefficients import * #  ymeans_sw_absorption, ysigma_sw_absorption, \
   # ymeans_sw_ray, ysigma_sw_ray, ymeans_lw_absorption, ysigma_lw_absorption

from ml_eval_funcs import plot_hist2d, plot_hist2d_T
from ml_trainfuncs_keras import create_model_mlp, savemodel




# ----------------------------------------------------------------------------
# ----------------- Provide data with inputs and outputs ---------------------
# ----------------------------------------------------------------------------

datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/peter/data/"

fpath_test  = datadir+"RRTMGP_data_g224_CAMS_2015_RND.nc"

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

use_existing_scaling_coefficients = True

# Model training: use CPU or GPU?
use_gpu = False

x_test_raw,y_test_raw,col_dry_test  = load_rrtmgp(fpath_test, predictand)


# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
# Input scaling - min-max

if use_existing_scaling_coefficients:
    # xmin = np.array([1.60E2, 5.15E-3, 1.01E-2, 4.36E-3, 1.41E-4, 0.00E0, 2.55E-8], dtype=np.float32)
    # xmax = np.array([ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3,
    #          5.8135214E-7, 3.6000001E-6], dtype=np.float32) 
    # xcoeffs = (xmin,xmax)
    
    # Output scaling 
    nfac = 8 # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
    # log scaling. Useful when the output is a vector which has a wide range 
    # of magnitudes across the vector elements (g-points)
    # After this, use standard-scaling
    xmin = xmin_all
    xmax = xmax_all
    input_names = input_names_all
    if (predictand == 'sw_absorption'):
        ymean  = ymeans_sw_absorption_224
        ystd   = ysigma_sw_absorption_224
        xmin = xmin_all[0:7]; xmax = xmax_all[0:7]
        input_names = input_names_all[0:7]

else:
    print("code missing")

xcoeffs = (xmin,xmax)

# test
x_test,y_test   = scale_gasopt(x_test_raw, y_test_raw, col_dry_test, scale_inputs, 
        scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)

x_test[x_test<0.0] = 0.0


from keras.models import load_model
fpath_keras = '../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.h5'

model = load_model(fpath_keras,compile=False)


ntest = 60*1800

x_test = x_test[0:ntest,:]
y_test = y_test[0:ntest,:]


start = time.time()
y_pred       = model.predict(x_test); 
end = time.time()
print(end - start) 



# MODEL SAVING; first as TensorFlow SaveModel
fpath = 'tmp/saved_model/'
fpath_onnx = fpath+".onnx"
# 
model.save(fpath, save_format='tf')
# newmodel.save(fpath, save_format='tf',save_traces=False)

# Now convert to ONNX model
os.system("python -m tf2onnx.convert --saved-model {} --output {} --opset 13".format(fpath,fpath_onnx)) 



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
import onnxruntime as ort

# Adjust session options
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession(fpath_onnx, sess_options=opts)



start = time.time()

y_pred = sess.run(["dense_2"], {"dense_input": x_test})[0]
end = time.time()
print(end - start)