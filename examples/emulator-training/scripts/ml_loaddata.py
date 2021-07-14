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

# input scaling coefficients for RRTMGP-NN - these should probably be put in an
# external file 
ymeans_sw_abs = np.array([3.64390580e-04, 4.35663940e-04, 4.98018635e-04, 5.77545608e-04,
       6.80800469e-04, 7.98740832e-04, 9.35279648e-04, 1.16656872e-03,
       1.58452173e-03, 1.86584354e-03, 1.99465151e-03, 2.16701976e-03,
       2.41802959e-03, 2.82146805e-03, 3.48183908e-03, 4.09035478e-03,
       3.24113556e-04, 3.74707161e-04, 4.17389121e-04, 4.57456830e-04,
       4.98836453e-04, 5.49621007e-04, 6.13025972e-04, 7.00094330e-04,
       8.49446864e-04, 9.81244841e-04, 1.03521883e-03, 1.10830076e-03,
       1.21134310e-03, 1.31195760e-03, 1.39195414e-03, 1.45876186e-03,
       4.28469037e-04, 5.20646572e-04, 6.22550666e-04, 7.22033263e-04,
       8.23189737e-04, 9.40993312e-04, 1.06700219e-03, 1.21110224e-03,
       1.45994173e-03, 1.69180706e-03, 1.79443962e-03, 1.92319078e-03,
       2.08631344e-03, 2.33873748e-03, 2.59446981e-03, 2.72375043e-03,
       2.89453164e-04, 3.26674141e-04, 3.59543861e-04, 3.93101625e-04,
       4.30800777e-04, 4.71213047e-04, 5.19042369e-04, 5.83244429e-04,
       6.85371691e-04, 7.79234222e-04, 8.19451292e-04, 8.65648268e-04,
       9.23064887e-04, 1.00047945e-03, 1.08587136e-03, 1.09644048e-03,
       2.97961291e-04, 3.59470578e-04, 4.12600290e-04, 4.63586446e-04,
       5.14341518e-04, 5.67600771e-04, 6.28228823e-04, 7.09333457e-04,
       8.51527497e-04, 9.86502739e-04, 1.04738004e-03, 1.12565351e-03,
       1.23939372e-03, 1.37201988e-03, 1.52829266e-03, 1.63304247e-03,
       2.70115474e-04, 3.13259574e-04, 3.55943455e-04, 4.24126105e-04,
       5.12095110e-04, 5.70286124e-04, 6.33014424e-04, 7.20241107e-04,
       8.71218799e-04, 1.01254229e-03, 1.07443938e-03, 1.14905764e-03,
       1.24553905e-03, 1.37227518e-03, 1.50288385e-03, 1.59471075e-03,
       2.49097910e-04, 3.04214394e-04, 3.61522223e-04, 4.22842306e-04,
       4.84134798e-04, 5.45158167e-04, 6.10073039e-04, 6.93159061e-04,
       8.35262705e-04, 9.70904832e-04, 1.03256141e-03, 1.10617711e-03,
       1.19320781e-03, 1.30544091e-03, 1.43013091e-03, 1.53011072e-03,
       3.03399313e-04, 3.29578412e-04, 3.45331355e-04, 3.61360639e-04,
       3.78781464e-04, 3.96694435e-04, 4.13389760e-04, 4.41241835e-04,
       5.20797505e-04, 6.10202318e-04, 6.68037275e-04, 7.32506509e-04,
       7.93701038e-04, 8.53195263e-04, 9.09500406e-04, 9.46514192e-04,
       2.32592269e-04, 2.71835335e-04, 3.08167830e-04, 3.44223663e-04,
       3.79800709e-04, 4.20017983e-04, 4.63830249e-04, 5.07036340e-04,
       5.73535392e-04, 6.31669944e-04, 6.62838283e-04, 7.10128399e-04,
       7.75139430e-04, 8.65899900e-04, 9.70544817e-04, 1.06985983e-03,
       3.41864652e-04, 3.69071873e-04, 3.96323914e-04, 4.22754820e-04,
       4.46886756e-04, 4.71655425e-04, 4.98428126e-04, 5.32940670e-04,
       6.06888614e-04, 6.70523092e-04, 7.07189436e-04, 7.71613559e-04,
       8.81720975e-04, 1.09840697e-03, 1.38371368e-03, 1.53473706e-03,
       4.17014409e-04, 4.30032291e-04, 4.34701447e-04, 4.33250068e-04,
       4.38496791e-04, 4.55915550e-04, 4.31207794e-04, 4.22994781e-04,
       4.37038223e-04, 4.48761188e-04, 4.56356967e-04, 4.66349302e-04,
       4.80149902e-04, 4.99643327e-04, 5.27186319e-04, 5.57484163e-04,
       2.16507342e-05, 2.14800675e-05, 2.42409842e-05, 2.30929109e-05,
       2.50050962e-05, 2.49029163e-05, 2.54020069e-05, 2.31895119e-05,
       2.51217079e-05, 2.50334833e-05, 2.48085526e-05, 2.50862649e-05,
       2.36565447e-05, 2.40919053e-05, 2.22349281e-05, 2.54304305e-05,
       4.07712301e-04, 5.41551271e-04, 6.77760807e-04, 8.20003042e-04,
       9.50566900e-04, 1.05036958e-03, 1.11506274e-03, 1.15274324e-03,
       1.17376540e-03, 1.18031248e-03, 1.18122390e-03, 1.18179375e-03,
       1.18207792e-03, 1.18228467e-03, 1.18242903e-03, 1.18247792e-03,
       1.02293247e-03, 1.05753809e-03, 1.11542153e-03, 1.18556281e-03,
       1.24850264e-03, 1.29191973e-03, 1.30981358e-03, 1.31571468e-03,
       1.31824624e-03, 1.31927885e-03, 1.31954963e-03, 1.31999550e-03,
       1.32057443e-03, 1.32077874e-03, 1.32089714e-03, 1.32086105e-03])
ysigma_sw_abs = np.repeat(0.00065187697,224)


def load_inp_outp_rrtmgp(fname,predictand, dcol=1, skip_lastlev=False):
    # Load data for training a GAS OPTICS (RRTMGP) emulator,
    # where inputs are layer-wise atmospheric conditions (T,p, gas concentrations)
    # and outputs are vectors of optical properties across g-points (e.g. optical depth)
    dat = Dataset(fname)
    
    if predictand not in ['tau_lw', 'planck_frac', 'tau_sw', 'tau_sw_abs', 
                          'tau_sw_ray', 'ssa_sw']:
        sys.exit("Second drgument to load_inp_outp_rrtmgp (predictand) " \
        "must be either tau_lw, planck_frac, tau_sw, tau_sw_abs, tau_sw_ray, or ssa_sw")
            
    # inputs
    if predictand in ["tau_lw", "planck_frac"]: # Longwave
        x = dat.variables['rrtmgp_lw_input'][:].data
    else: # Shortwave
        x = dat.variables['rrtmgp_sw_input'][:].data
    nx = x.shape[-1]
    
    # outputs
    if (predictand=='tau_sw_ray'):
        ssa = dat.variables['ssa_sw'][:].data
        tau = dat.variables['tau_sw'][:].data
        y = tau * ssa # tau_sw_ray = tau_tot * single scattering albedo
        del tau, ssa
    elif (predictand=='tau_sw_abs'):
        ssa = dat.variables['ssa_sw'][:].data
        tau = dat.variables['tau_sw'][:].data
        tau_sw_ray = tau * ssa
        y = tau - tau_sw_ray # tay_sw_abs = tau_tot - tau_ray
        del tau, ssa, tau_sw_ray
    else:
        y  = dat.variables[predictand][:].data
        
    # if predictand in ['tau_lw','tau_sw', 'ssa_sw']:
    col_dry = dat.variables['col_dry'][:].data
    
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
        col_dry = col_dry[:,0:-1,:]
    if dcol>1:
        y  = y[:,::dcol,:,:]; x  = x[:,::dcol,:,:]
        col_dry = col_dry[:,::dcol,:]

    nobs = nexp*ncol*nlay
    print( "there are {} profiles (expt*col) this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))

    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,nx))
    col_dry = np.reshape(col_dry,(nobs))
    
    return x,y,col_dry

def load_inp_outp_rte_rrtmgp_sw(fname, predictand, clouds=True):
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gasopt = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    (nexp,ncol,nlay,nx) = x_gasopt.shape
    # plus surface albedo, which !!!FOR THIS DATA!!! is spectrally constant
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol,ngpt)
    sfc_alb = sfc_alb[:,:,0] # (nexp,ncol)
    # plus by cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # # ..multiplied by incoming flux
    # #  (ASSUMED CONSTANT)
    # toa_flux = dat.variables['toa_flux'][:].data # (nexp,ncol,ngpt)
    # ngpt = toa_flux.shape[-1]
    # for iexp in range(nexp):
    #     for icol in range(ncol):
    #         toa_flux[iexp,icol,:] = mu0[iexp,icol] * toa_flux[iexp,icol,:]
    if clouds:
        ciwc = dat.variables['ciwc'][:].data
        clwc = dat.variables['clwc'][:].data

    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    y0 = dat.variables['rsu'][:]
    y1 = dat.variables['rsd'][:]
        
    if np.size(y0.shape) == 3:
        (nexp,ncol,nlev) = y0.shape
    elif np.size(y0.shape) == 2:
        (ncol,nlev) = y0.shape
        nexp = 1
        y0 = np.reshape(y0,(nexp,ncol,nlev)); y1 = np.reshape(y1,(nexp,ncol,nlev))
        x_gasopt = np.reshape(x_gasopt,(nexp,ncol,nlay,nx))
    else:
        sys.exit("Invalid array shapes, RTE output should have 2 or 3 dimensions")
    
    # Reshape to profiles...
    ns = nexp*ncol # number of samples (profiles)
    y  = np.zeros((ns,nlev*2))
    # inputs: one input vector consists of the atmospheric profile (T,p,gases),
    # plus mu0 (1) plus surface albedo (1)...these need to be flattened and
    # stacked on top of each other
    x_gasopt = np.reshape(x_gasopt,(nexp,ncol,nlay*nx)) # to profiles
    nx_gasopt = nlay*nx
    # nx_rte = ngpt + 1 # inc flux + albedo
    nx_rte = 1 + 1 # solar angle + albedo
    nx_clouds = 0
    if clouds: nx_clouds = 2*nlay
    x  = np.zeros((ns,(nx_gasopt + nx_rte + nx_clouds)))
    # need to reshape mu0 and sfc alb so they are also 3D
    mu0     = np.reshape(mu0,(nexp,ncol,1))
    sfc_alb = np.reshape(sfc_alb,(nexp,ncol,1))
    
    i = 0
    if clouds:
        for iexp in range(nexp):
           for icol in range(ncol):
               y[i,:] = np.concatenate((y0[iexp,icol,:], y1[iexp,icol,:]))
               x[i,:] = np.concatenate((x_gasopt[iexp,icol,:], mu0[iexp,icol], 
                        sfc_alb[iexp,icol], ciwc[iexp,icol,:], clwc[iexp,icol,:]))
               i = i + 1
    else:
        for iexp in range(nexp):
           for icol in range(ncol):
               y[i,:] = np.concatenate((y0[iexp,icol,:], y1[iexp,icol,:]))
               x[i,:] = np.concatenate((x_gasopt[iexp,icol,:], mu0[iexp,icol], sfc_alb[iexp,icol]))
               i = i + 1
           
    print( "there are {} profiles (expt*col) this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    return x,y


def load_inp_outp_rte_sw(fname):
    # Load data for training a RADIATIVE TRANSFER  SOLVER (RTE) emulator,
    # where inputs are vertical PROFILES of optical properties (tau, ssa, g)
    # + boundary conditions, for one spectral point (g-point)
    # outputs (predictand) are PROFILES of broadband fluxes (up and down)
    # also per spectral point
    
    dat = Dataset(fname)
    
    tau = dat.variables['tau_sw'][:].data # nexp, ncol, nlay, ngpt
    ssa = dat.variables['ssa_sw'][:].data # nexp, ncol, nlay, ngpt
    g = dat.variables['g_sw'][:].data # nexp, ncol, nlay, ngpt
    
    (nexp,ncol,nlay,ngpt) = tau.shape
    # plus surface albedo, which !!!FOR THIS DATA!!! is spectrally constant
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol,ngpt)
    sfc_alb = sfc_alb[:,:,0] # (nexp,ncol)
    # plus by cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # # ..multiplied by incoming flux
    # #  (ASSUMED CONSTANT)
    # toa_flux = dat.variables['toa_flux'][:].data # (nexp,ncol,ngpt)
    # ngpt = toa_flux.shape[-1]
    # for iexp in range(nexp):
    #     for icol in range(ncol):
    #         toa_flux[iexp,icol,:] = mu0[iexp,icol] * toa_flux[iexp,icol,:]


    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    y0 = dat.variables['rsu'][:] # (nexp,ncol,nlev,ngpt)
    y1 = dat.variables['rsd'][:]
    
    nlev = nlay+1
        
    # Permute to (nexp,ncol,ngpt,nlev)
    y0 = np.swapaxes(y0,2,3)
    y1 = np.swapaxes(y1,2,3)
    if (y0.shape[-1] != nlev):
        sys.exit("Invalid shape for gpt flux, last dimension after permuting should be nlev")
    tau = np.swapaxes(tau,2,3)
    ssa = np.swapaxes(ssa,2,3)
    g = np.swapaxes(g,2,3)
    if (tau.shape[-1] != nlay):
        sys.exit("Invalid shape for tau, last dimension after permuting should be nlay")

    # Reshape to 2D data matrix 
    ns = nexp*ncol*ngpt # number of samples 
    y  = np.zeros((ns,nlev*2))
    # inputs: one input vector consists of vertical profiles of tau+ssa+g,
    # plus mu0 and surface albedo..these variables all need to be flattened and
    # stacked on top of each other
    x = np.zeros(ns,(3*nlay + 1 + 1))
    # need to reshape mu0 and sfc alb so they are 1D arrays of size 1 when indexed
    mu0     = np.reshape(mu0,(nexp,ncol,1))
    sfc_alb = np.reshape(sfc_alb,(nexp,ncol,1))
    
    i = 0
    for iexp in range(nexp):
       for icol in range(ncol):
           for igpt in range(ngpt):
               y[i,:] = np.concatenate((y0[iexp,icol,igpt,:], y1[iexp,icol,igpt,:]))
               x[i,:] = np.concatenate((tau[iexp,icol,igpt], ssa[iexp,icol,igpt],
                g[iexp,icol,igpt], mu0[iexp,icol], sfc_alb[iexp,icol]))
               i = i + 1
           
    print( "there are {} profiles (expt*col) this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    return x,y

def preproc_tau_to_crossection(tau, col_dry):
    y = np.zeros(tau.shape)
    for igpt in range(tau.shape[1]):
        y[:,igpt]  = tau[:,igpt] / col_dry
    return y

@njit(parallel=True)
def preproc_pow_gptnorm(y, nfac, means,sigma):
    # scale y to y', where y is a data matrix of shape (nsamples, ng) consisting
    # consisting of ng outputs; e.g. g-point vector of absorption cross-sections,
    # and y' has been scaled for more effective neural network training
    # y is first power-scaled by y = y**(1/nfac) and then normalized by
    # y'g =  (y_g - mean(y_g)) / std(y_g), where g is a single g-point
    # using means of individual g-points but sigma across g-points 
    # is recommended as it preserves correlations but scales to a common range
    # the means and sigma(s) are input arguments as they need to be fixed 
    # for production
    (nobs,ngpt) = y.shape
    y_scaled = np.zeros(y.shape)
    nfacc = 1/nfac
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y_scaled[iobs,igpt] = np.power(y[iobs,igpt],nfacc)
            y_scaled[iobs,igpt] = (y_scaled[iobs,igpt] - means[igpt]) / sigma[igpt]
    return y_scaled

@njit(parallel=True)
def preproc_pow_gptnorm_reverse(y_scaled, nfac, means,sigma):
    # y has shape (nobs,gpts)
    y = np.zeros(y_scaled.shape)

    (nobs,ngpt) = y_scaled.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y_scaled[iobs,igpt] * sigma[igpt]) + means[igpt]
            y[iobs,igpt] = np.power(y[iobs,igpt],nfac)

    return y

    