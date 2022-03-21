#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of RRTMGP gas optics
scheme, the RTE radiative transfer solver, or their combination RTE+RRTMGP (a 
radiative transfer scheme).

This file provides functions for loading and preprocessing data so that it
may be used for training e.g. a neural network version of RRTMGP

@author: Peter Ukkonen
"""

import os, subprocess, argparse
import sys
import numpy as np
from numba import jit, njit, prange
from netCDF4 import Dataset
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

def save_model_netcdf(fpath_netcdf, model, activation_names, input_names, 
                    emulator_target,
                    xmin, xmax, ymean=None, ystd=None, y_scaling_comment=None, 
                    x_scaling_comment=None, data_comment=None, model_comment=None):
    # Model = Keras Sequential Model object describing a Dense NN
    # activation_names = numpy array of strings describing the activation function used
    #   function used in each hidden+output layer (e.g. ['relu', 'linear'] - 
    #   these need to correspond to the names in Neural-Fortran
    # input_names = numpy array of strings containing names of inputs - 
    #   the gas names need to correspond to the names used in RRTMGP-NN
    # xmin, xmax = numpy arrays containing the min/max values of the inputs,  
    #   required for pre-processing
    #  OPTIONAL ARGUMENTS - if present, assumed predictand is absorbtion/Rayleigh
    #  cross-sections, otherwise Planck fraction
    # ymean, ystd = numpy arrays containing the mean and standard deviation of
    #   outputs, required for post-processing
    
    from netCDF4 import Dataset, stringtochar
    
    # Create a netCDF file specifying the NN. in case of two hidden layers it will
    # look like this:
    # dimensions
    #   nn_layers  : 3        <--- does not include the input layer
    #   nn_dim_inp    : 7     <--- nn_dim_* specifies the length of a dimension
    #   nn_dim_hidden1: 16
    #   nn_dim_hidden2: 16
    #   nn_dim_outp   : 224
   #
    # variables:
    #   integer nn_dimsize(layers) = [16,16,224]
    #   float nn_weights_1(nn_dim_inp,     nn_dim_hidden1)
    #   float nn_bias_1(nn_dim_hidden1)
    #   float nn_weights_2(nn_dim_hidden1, nn_dim_hidden2)
    #   float nn_bias_2(nn_dim_hidden2)
    #   float nn_weights_3(nn_dim_hidden2, nn_dim_outp)
    #   float nn_bias_3(nn_dim_outp)
    #   string nn_inputs(nn_dim_inp)
    #   string nn_activation(layers)    ...

    # Create a netCDF file
    dat_new     = Dataset(fpath_netcdf,'w')
    # Number of NN layers
    nlay = np.size(model.layers) 
    # Number of inputs
    nx = model.layers[0].get_weights()[0].shape[0]

    # Create initial dimensions
    dat_new.createDimension('nn_layers',nlay)

    # The first NN dimension corresponds to the input (but it's not a NN layer)
    str_dim_prev = 'nn_dim_input'
    dat_new.createDimension(str_dim_prev, nx)
    # Create initial variables
    nc_dimsize      = dat_new.createVariable("nn_dimsize","i4",("nn_layers"))
    nc_dimsize.long_name = "Dimension of each layer, not including the input layer"
    nc_activ        = dat_new.createVariable("nn_activation","str",("nn_layers"))
    nc_input        = dat_new.createVariable("nn_inputs","str",(str_dim_prev))
    nc_input.long_name = 'Specifies the inputs in their correct order'
    nc_input_coeffs_max   = dat_new.createVariable("nn_input_coeffs_max","f4",(str_dim_prev))
    nc_input_coeffs_min   = dat_new.createVariable("nn_input_coeffs_min","f4",(str_dim_prev))
    nc_input_coeffs_max.long_name = 'xmax, see global attribute input_scaling_info'
    nc_input_coeffs_min.long_name = 'xmin, see global attribute input_scaling_info'
   
    dat_new.emulator_target = emulator_target
    if not data_comment==None:
        dat_new.data_info = data_comment
    if not model_comment==None:
        dat_new.model_info = model_comment

        
    # NetCDF Fortran can't handle strings (ugh)
    dat_new.createDimension('string_len', 32)
    nc_activ_char   = dat_new.createVariable("nn_activation_char","S1",("nn_layers","string_len"))
    nc_input_char   = dat_new.createVariable("nn_inputs_char","S1",(str_dim_prev,"string_len"))

    # Write initial data
    nc_activ_char[:,:] = ' '
    nc_input_char[:,:] = ' '
    
    nc_input_coeffs_max[:] = xmax
    nc_input_coeffs_min[:] = xmin

    # loop over hidden layers + output layer, where each of these are associated
    # with weights, biases and activation - create the dimension and variables of
    # each layer
    # does not include the input layer as this should not be considered a NN layer 
    # according to Bishop's book (Pattern recognition and Machine Learning)
    for i in range(nlay):
        j = i+1

        weight = model.layers[i].get_weights()[0]
        bias = model.layers[i].get_weights()[1]
        
        dimsize         = weight.shape[1]
        
        # Create dimension corresponding to this layer
        if (i<nlay-1):
            str_dim_this    = 'nn_dim_hidden' + str(j)
        else:
            str_dim_this    = 'nn_dim_outp'
        dat_new.createDimension(str_dim_this,dimsize)

        
        # Create weight variable
        str_weight = 'nn_weights_' + str(j)
        str_bias  =  'nn_bias_' + str(j)
        nc_weight  = dat_new.createVariable(str_weight,"f4",(str_dim_prev,str_dim_this))
        nc_bias    = dat_new.createVariable(str_bias,  "f4",(str_dim_this))
                                               
        # Write the data
        nc_dimsize[i]   = dimsize
        nc_weight[:]    = weight
        nc_bias[:]      = bias
        # Write the activation function as a string
        activ_str       = activation_names[i]
        nc_activ[i]     = activ_str
        
        charfmt = "S{}".format(len(activ_str))
        activ_chars = stringtochar(np.array(activ_str, charfmt))
        
        nc_activ_char[i,0:len(activ_str)] = activ_chars
        
        str_dim_prev = str_dim_this

    if not np.any(ymean)==None:  
        nc_output_coeffs_mean   = dat_new.createVariable("nn_output_coeffs_mean","f4",('nn_dim_outp'))
        nc_output_coeffs_std    = dat_new.createVariable("nn_output_coeffs_std","f4",('nn_dim_outp'))
        nyy = ymean.size
        nc_output_coeffs_mean[0:nyy]    = ymean
        nc_output_coeffs_std[0:nyy]     = ystd
        nc_output_coeffs_mean.long_name = 'ymean(igpt) = mean(y_cross(igpt)**(1/8))'
        nc_output_coeffs_std.long_name = 'ystd(igpt) = std(y_cross(igpt)**(1/8))'

    if not y_scaling_comment==None:
        dat_new.output_scaling_info = y_scaling_comment
        
    for i in range(nx):
        input_str   = input_names[i]
        nc_input[i] = input_str
        
        charfmt = "S{}".format(len(input_str))
        input_chars = stringtochar(np.array(input_str, charfmt))
        
        nc_input_char[i,0:len(input_str)] = input_chars
    
    if not x_scaling_comment==None:
        dat_new.input_scaling_info = x_scaling_comment

    dat_new.close()



def load_rrtmgp(fname,predictand, dcol=1, skip_lastlev=False, skip_firstlev=False,expfirst=False):
    # Load data for training a GAS OPTICS (RRTMGP) emulator,
    # where inputs are layer-wise atmospheric conditions (T,p, gas concentrations)
    # and outputs are vectors of optical properties across g-points (e.g. optical depth)
    dat = Dataset(fname)
    
    if predictand not in ['lw_absorption', 'lw_planck_frac', 'sw_absorption', 
                          'sw_rayleigh', 'lw_both']: 
        sys.exit("Second drgument to load_rrtmgp (predictand) " \
        "must be either lw_absorption, lw_planck_frac, sw_absorption, or sw_rayleigh")
            
    # k-distribution info
    try: 
        kdist_str = dat.comment
        kdist_str = kdist_str.split(' ')
        kdist_str = kdist_str[2]
        kdist_str = kdist_str.split('/')
        kdist_str = kdist_str[4]
    except:
        kdist_str = None
        
    # inputs
    if predictand in ["lw_absorption", "lw_planck_frac",'lw_both']: # Longwave
        xname = 'rrtmgp_lw_input'
    else: # Shortwave
        xname = 'rrtmgp_sw_input'
        
    x = dat.variables[xname][:].data
    
    try:
        input_names = dat.variables[xname].comment
        print('input_names found in file')
        input_names = input_names.split(' ')
        if (input_names[0]=='Features:'):
            input_names = input_names[1:]
    except:
        print("input_names not found in file") 
        input_names = None
        
    nx = x.shape[-1]
    
    # outputs
    if (predictand=='sw_rayleigh'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
        y = tau * ssa # tau_sw_rayleigh = tau_tot * single scattering albedo
        del tau, ssa
    elif (predictand=='sw_absorption'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
        tau_sw_rayleigh = tau * ssa
        y = tau - tau_sw_rayleigh # tay_sw_abs = tau_tot - tau_ray
        del tau, ssa, tau_sw_rayleigh
    elif (predictand=='lw_absorption'):
        y = dat.variables['tau_lw_gas'][:].data
    elif (predictand=='lw_planck_frac'):
        y = dat.variables['planck_fraction'][:].data
    elif (predictand=='lw_both'):
        y = dat.variables['tau_lw_gas'][:].data
        y2 = dat.variables['planck_fraction'][:].data   
        y = np.concatenate((y,y2),axis=-1)
    else:
        y  = dat.variables[predictand][:].data
        
    # if predictand in ['lw_absorption','tau_sw', 'ssa_sw']:
    col_dry = dat.variables['col_dry'][:].data
    
    if col_dry[0,0,0] == 0.0:
        skip_firstlev = True
        
    if np.size(y.shape) == 4:
        (nexp,ncol,nlay,ngpt) = y.shape
    elif np.size(y.shape) == 3:
        (ncol,nlay,ngpt) = y.shape
        nexp = 1
        y = np.reshape(y,(nexp,ncol,nlay,ngpt))
        x = np.reshape(y,(nexp,ncol,nlay,nx))
        col_dry = np.reshape(col_dry,(nexp,ncol,nlay))
    else:
        sys.exit("Invalid array shapes, RRTMGP output should have at least 3 dimensions")
        
    if skip_lastlev:
        x = x[:,:,0:-1,:]; y = y[:,:,0:-1,:]
        col_dry = col_dry[:,:,0:-1]
        nlay = nlay -1
        
    if skip_firstlev:
        x = x[:,:,1:,:]; y = y[:,:,1:,:]
        col_dry = col_dry[:,:,1:]
        nlay = nlay - 1 
        
    if dcol>1:
        y  = y[:,::dcol,:,:]; x  = x[:,::dcol,:,:]
        col_dry = col_dry[:,::dcol,:]

    nobs = nexp*ncol*nlay
    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))

    if expfirst:
        print("Reshaping so that adjacent samples are from different experiments")
        x = np.rollaxis(x,0,3)     
        y = np.rollaxis(y,0,3) 
        col_dry = np.rollaxis(col_dry,0,3)
        
    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,nx))
    col_dry = np.reshape(col_dry,(nobs))
    
    return x,y,col_dry,input_names,kdist_str


def get_col_dry(vmr_h2o, plev):
    grav = 9.80665
    m_dry = 0.028964
    m_h2o =  0.018016
    avogad = 6.02214076e23
    delta_plev = plev[:,1:] - plev[:,0:-1]
    # Get average mass of moist air per mole of moist air
    fact = 1.0 / (1. + vmr_h2o)
    m_air = (m_dry + m_h2o * vmr_h2o) * fact
    col_dry = 10.0 * np.float64(delta_plev) * avogad * np.float64(fact) / (1000.0 * m_air * 100.0 * grav)
    return np.float32(col_dry)

@njit(parallel=True)
def preproc_tau_to_crossection(tau, col_dry):
    y = np.zeros(tau.shape,dtype=np.float32)
    for iobs in range(tau.shape[0]):
        for igpt in range(tau.shape[1]):
            y[iobs,igpt]  = tau[iobs,igpt] / col_dry[iobs]
    return y

@njit(parallel=True)
def preproc_pow_standardization(y, nfac, means,sigma):
    # scale y to y', where y is a data matrix of shape (nsamples, ng) consisting
    # consisting of ng outputs; e.g. g-point vector of absorption cross-sections,
    # and y' has been scaled for more effective neural network training
    # y is first power-scaled by y = y**(1/nfac) and then normalized by
    # y'g =  (y_g - mean(y_g)) / std(y_g), 
    # when training correlated-k gas optics models, g is a single g-point.
    # using means of individual g-points but sigma across g-points 
    # is recommended as it preserves correlations but scales to a common range
    # the means and sigma(s) are input arguments as they need to be fixed 
    # for production
    (nobs,ngpt) = y.shape
    y_scaled = np.zeros(y.shape,dtype=np.float32)

    nfacc = 1/nfac
    for iobs in prange(nobs):
       for igpt in prange(ngpt):
           y_scaled[iobs,igpt] = np.power(y[iobs,igpt],nfacc)
           y_scaled[iobs,igpt] = (y_scaled[iobs,igpt] - means[igpt]) / sigma[igpt]
                
    return y_scaled



@njit(parallel=True)
def preproc_pow_standardization_reverse(y_scaled, nfac, means,sigma):
    # y has shape (nobs,gpts)
    y = np.zeros(y_scaled.shape)

    (nobs,ngpt) = y_scaled.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y_scaled[iobs,igpt] * sigma[igpt]) + means[igpt]
            y[iobs,igpt] = np.power(y[iobs,igpt],nfac)

    return y

@njit(parallel=True)
def preproc_standardization(y, means,sigma):
    (nobs,ngpt) = y.shape
    y_scaled = np.copy(y)
    for iobs in prange(nobs):
       for igpt in prange(ngpt):
           y_scaled[iobs,igpt] = (y_scaled[iobs,igpt] - means[igpt]) / sigma[igpt]
                
    return y_scaled

@njit(parallel=True)
def preproc_standardization_reverse(y_scaled, means,sigma):
    # y has shape (nobs,gpts)
    y = np.zeros(y_scaled.shape)
    (nobs,ngpt) = y_scaled.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y_scaled[iobs,igpt] * sigma[igpt]) + means[igpt]
            
    return y

def preproc_minmax_inputs(x, xcoeffs=None):
        x_scaled = np.copy(x)
        if xcoeffs is None:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()  
            scaler.fit(x_scaled)
            x_scaled = scaler.transform(x_scaled)  
            return x_scaled, scaler.data_min_, scaler.data_max_
        else:
            (xmin,xmax) = xcoeffs
            for i in range(x.shape[1]):
                if (xmax[i] - xmin[i]) == 0.0:
                    x_scaled[:,i] = 0.0
                else:
                    x_scaled[:,i] =  (x_scaled[:,i] - xmin[i]) / (xmax[i] - xmin[i] )
            return x_scaled

def preproc_divbymax(x,xmax=None):
    x_scaled = np.copy(x)
    if xmax is None:
        xmax = np.zeros(x.shape[-1])
        
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,:,i])
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                xmax[i] =  np.max(x[:,i])
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
        return x_scaled, xmax
    else:
        if np.size(x.shape)==3:
            for i in range(x.shape[-1]):
                x_scaled[:,:,i] =  x_scaled[:,:,i] / xmax[i]
        else:
            for i in range(x.shape[-1]):
                x_scaled[:,i] =  x_scaled[:,i] / xmax[i]
                
        return x_scaled

def preproc_minmax_reverse(x_scaled, xcoeffs):
        x = np.copy(x_scaled)

        (xmin,xmax) = xcoeffs
        for i in range(x.shape[1]):
            if (xmax[i] - xmin[i]) == 0.0:
                x[:,i] = 0.0
            else:
                x[:,i] =  (x[:,i] + xmin[i]) * (xmax[i] - xmin[i] )
        return x

# Preprocess RRTMGP inputs (p,T, gas concs)
def preproc_minmax_inputs_rrtmgp(x, xcoeffs=None): #, datamin, datamax):
        x_scaled = np.copy(x)
        # Log-scale pressure, power-scale H2O and O3
        x_scaled[:,1] = np.log(x_scaled[:,1])
        x_scaled[:,2] = x_scaled[:,2]**(1.0/4) 
        x_scaled[:,3] = x_scaled[:,3]**(1.0/4) 
        # x = minmaxscale(x,data_min_,data_max_)
        if xcoeffs==None:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()  
            scaler.fit(x_scaled)
            x_scaled = scaler.transform(x_scaled) 
            return x_scaled, scaler.data_min_, scaler.data_max_

        else:
            (xmin,xmax) = xcoeffs
            for i in range(x.shape[1]):
                x_scaled[:,i] =  (x_scaled[:,i] - xmin[i]) / (xmax[i] - xmin[i] )
            return x_scaled

# A wrapping function to scale inputs and/or outputs   
def scale_gasopt(x_raw, y_raw, col_dry, scale_inputs=False, scale_outputs=False, 
                 nfac=1, y_mean=None, y_sigma=None, xcoeffs=None):

    if scale_inputs:
        if xcoeffs is None:
            x,xmin,xmax = preproc_minmax_inputs_rrtmgp(x_raw)
        else:
            x = preproc_minmax_inputs_rrtmgp(x_raw,xcoeffs )
    else:
        x = x_raw
        
    if scale_outputs:
        # Standardization coefficients loaded from file
        #  y_mean = ymeans_sw_abs; y_sigma = ysigma_sw_abs
        # Set power scaling coefficient (y == y**(1/nfac))
        # nfac = 8 
        if np.any(y_sigma)==None:
            y_sigma = np.repeat(np.float32(1),y_raw.shape[1])

        if np.any(y_mean)==None:
            y_mean = np.repeat(np.float32(0),y_raw.shape[1])
        
        # Scale by layer number of molecules to obtain absorption cross section
        y   = preproc_tau_to_crossection(y_raw, col_dry)
        # Scale using power-scaling followed by standard-scaling
        y   = preproc_pow_standardization(y, nfac, y_mean, y_sigma)
    else:
        y = y_raw
        
    if xcoeffs is None:
        return x, y, xmin, xmax
    else: return x,y
    
def scale_outputs_wrapper(y_raw, col_dry, predictand, ymean=None, ystd=None):
    ny = y_raw.shape[1]
    if (predictand == 'lw_planck_frac'):
        nfac = 2
        y    = scale_outputs(y_raw, None, nfac, None, None)
        
    elif (predictand == 'lw_both'): 
        # I tested just having a unified LW model, didn't seem very promising
        # nfac = 4
        # nyy = int(ny/2)
        # y = y_raw.copy()
        # y[:,0:nyy] = preproc_tau_to_crossection(y[:,0:nyy], col_dry)

        # if np.any(ymean)==None:
        #     ymean = np.zeros(ny); ystd = np.zeros(ny)
        #     for i in range(ny):
        #         ymean[i] = np.mean(y[:,i]**(1/nfac))
        #         ystd[i]  = np.std(y[:,i]**(1/nfac))
        
        # # Scale data
        # y[:,0:nyy]   = scale_outputs(y_raw[:,0:nyy], col_dry, nfac, ymean[0:nyy], ystd[0:nyy])
        # y[:,nyy:]    = scale_outputs(y_raw[:,nyy:], None, nfac, ymean[nyy:], ystd[nyy:])
        
        nfac = 8
        nfac2 = 2
        nyy = int(ny/2)
        y = y_raw.copy()
        y[:,0:nyy] = preproc_tau_to_crossection(y[:,0:nyy], col_dry)

        if np.any(ymean)==None:
            ymean = np.zeros(nyy); ystd = np.zeros(nyy)
            for i in range(nyy):
                ymean[i] = np.mean(y[:,i]**(1/nfac))
                ystd[i]  = np.std(y[:,i]**(1/nfac))
        
        # Scale data
        y[:,0:nyy]   = scale_outputs(y_raw[:,0:nyy], col_dry, nfac, ymean[0:nyy], ystd[0:nyy])
        y[:,nyy:]    = scale_outputs(y_raw[:,nyy:], None, nfac2, None, None)
    else:  # For scaling optical depths
        nfac = 8

        y   = preproc_tau_to_crossection(y_raw, col_dry)

        if np.any(ymean)==None:
            ymean = np.zeros(ny); ystd = np.zeros(ny)
            for i in range(ny):
                ymean[i] = np.mean(y[:,i]**(1/nfac))
                # ystd[i]  = np.std(y[:,i]**(1/nfac))
            ystd = np.repeat(np.std(y**(1/nfac)),ny)
                
        # Scale data
        y    = scale_outputs(y_raw, col_dry, nfac, ymean, ystd)
    return y, ymean, ystd
        
def scale_outputs(y_raw, col_dry=None, nfac=1, 
                 y_mean=None, y_sigma=None):
    # Y_mean and y_sigma are optional outputs: if missing, skip standard-scaling
    if np.any(y_sigma)==None:
        y_sigma = np.repeat(np.float32(1), y_raw.shape[1])

    if np.any(y_mean)==None:
        y_mean = np.repeat(np.float32(0), y_raw.shape[1])
    
    # Scale by layer number of molecules to obtain absorption cross section
    if np.any(col_dry)==None:
        y = np.copy(y_raw)
    else:
        y = preproc_tau_to_crossection(y_raw, col_dry)

    # Scale using power-scaling followed by standard-scaling
    y   = preproc_pow_standardization(y, nfac, y_mean, y_sigma)
    return y