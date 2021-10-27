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
ymeans_sw_ray = np.array([0.00016408, 0.00016821, 0.00016852, 0.00016616, 0.0001631 ,
       0.0001615 , 0.00016211, 0.00016632, 0.00017432, 0.00017609,
       0.00017617, 0.00017683, 0.00017806, 0.00017891, 0.00017938,
       0.00017905, 0.00020313, 0.000203  , 0.00020417, 0.00020546,
       0.00020597, 0.00020647, 0.0002067 , 0.0002069 , 0.00020719,
       0.00020752, 0.00020766, 0.00020783, 0.00020801, 0.00020828,
       0.00020884, 0.00020988, 0.00022147, 0.00022575, 0.00022777,
       0.00022846, 0.00022824, 0.00022803, 0.00022816, 0.00022841,
       0.0002286 , 0.00022876, 0.00022883, 0.00022887, 0.00022888,
       0.00022891, 0.00022922, 0.00023004, 0.00025017, 0.00024942,
       0.00024824, 0.00024734, 0.00024655, 0.00024587, 0.00024539,
       0.00024486, 0.00024454, 0.0002441 , 0.00024381, 0.00024343,
       0.00024307, 0.00024265, 0.00024179, 0.00024042, 0.00025942,
       0.00026145, 0.00026296, 0.00026379, 0.00026454, 0.00026518,
       0.00026548, 0.00026566, 0.00026578, 0.00026592, 0.00026607,
       0.00026617, 0.00026612, 0.00026633, 0.00026634, 0.00026667,
       0.00028838, 0.00028652, 0.00028487, 0.00028311, 0.00027978,
       0.00027901, 0.00027885, 0.00027866, 0.000278  , 0.00027733,
       0.00027734, 0.00027694, 0.00027574, 0.00027526, 0.0002754 ,
       0.00027594, 0.00030356, 0.00031213, 0.00031563, 0.00031476,
       0.00031548, 0.00031693, 0.00031758, 0.00031764, 0.00031757,
       0.00031747, 0.0003175 , 0.0003176 , 0.00031767, 0.00031787,
       0.00031921, 0.00032022, 0.00033161, 0.00033401, 0.00033486,
       0.0003345 , 0.00033421, 0.00033408, 0.00033402, 0.00033377,
       0.00033366, 0.00033365, 0.0003337 , 0.0003337 , 0.00033372,
       0.00033368, 0.0003336 , 0.00033342, 0.00038102, 0.00038465,
       0.00038598, 0.00038884, 0.00039175, 0.00039174, 0.00039248,
       0.00039248, 0.00039339, 0.00038445, 0.00037872, 0.00037472,
       0.00037253, 0.00036779, 0.00036238, 0.00035383, 0.0004347 ,
       0.00044431, 0.00045121, 0.00045676, 0.00046053, 0.00046292,
       0.00046443, 0.00046354, 0.00045597, 0.0004476 , 0.00044526,
       0.00044227, 0.00043994, 0.00043726, 0.00043139, 0.00043131,
       0.00050991, 0.0005198 , 0.00052429, 0.00052806, 0.00052854,
       0.00052917, 0.00053388, 0.00053857, 0.0005396 , 0.00053686,
       0.00053488, 0.0005327 , 0.00052904, 0.00052511, 0.00052043,
       0.00051689, 0.00057637, 0.0005886 , 0.0006002 , 0.00061095,
       0.00062064, 0.00062908, 0.00063611, 0.00064162, 0.00064557,
       0.00064733, 0.00064765, 0.0006479 , 0.0006481 , 0.00064824,
       0.00064831, 0.00064833, 0.00065669, 0.00067188, 0.00068705,
       0.00070096, 0.00071349, 0.00072444, 0.00073358, 0.00074076,
       0.00074614, 0.00074835, 0.00074795, 0.00074786, 0.0007479 ,
       0.00074804, 0.00074818, 0.00074823, 0.0008143 , 0.00081451,
       0.00081412, 0.00081407, 0.00081408, 0.00081299, 0.00081158,
       0.00081498, 0.00081472, 0.0008145 , 0.00081539, 0.00081426,
       0.00081398, 0.000814  , 0.00081404, 0.00081415])
ysigma_sw_ray = np.repeat(0.00019679657,224)

def load_rrtmgp(fname,predictand, dcol=1, skip_lastlev=False):
    # Load data for training a GAS OPTICS (RRTMGP) emulator,
    # where inputs are layer-wise atmospheric conditions (T,p, gas concentrations)
    # and outputs are vectors of optical properties across g-points (e.g. optical depth)
    dat = Dataset(fname)
    
    if predictand not in ['tau_lw', 'planck_frac', 'tau_sw', 'tau_sw_abs', 
                          'tau_sw_ray', 'ssa_sw']:
        sys.exit("Second drgument to load_rrtmgp (predictand) " \
        "must be either tau_lw, planck_frac, tau_sw, tau_sw_abs, tau_sw_ray, or ssa_sw")
            
    # inputs
    if predictand in ["tau_lw", "planck_frac"]: # Longwave
        x = dat.variables['rrtmgp_lw_input'][:].data
    else: # Shortwave
        x = dat.variables['rrtmgp_sw_input'][:].data
    nx = x.shape[-1]
    
    # outputs
    if (predictand=='tau_sw_ray'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
        y = tau * ssa # tau_sw_ray = tau_tot * single scattering albedo
        del tau, ssa
    elif (predictand=='tau_sw_abs'):
        ssa = dat.variables['ssa_sw_gas'][:].data
        tau = dat.variables['tau_sw_gas'][:].data
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
    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))

    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,nx))
    col_dry = np.reshape(col_dry,(nobs))
    
    return x,y,col_dry

def load_radscheme(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, clouds=True, 
                   return_pressures=False, add_coldry=False):
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    # argument scale_p_h2o_o3 determines whether specific gas optics inputs
    # (pressure, H2O and O3 )  are power-scaled similarly to Ukkonen 2020 paper
    # for a more normal distribution
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gasopt = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    
    (nexp,ncol,nlay,nx) = x_gasopt.shape
    nlev = nlay+1
    ns = nexp*ncol # number of samples (profiles)
    if add_coldry:
        vmr_h2o = x_gasopt[:,:,:,2].reshape(ns,nlay)
    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gasopt[:,:,:,1] = np.log(x_gasopt[:,:,:,1])
        x_gasopt[:,:,:,2] = x_gasopt[:,:,:,2]**(1.0/4) 
        x_gasopt[:,:,:,3] = x_gasopt[:,:,:,3]**(1.0/4)

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
        lwp = dat.variables['cloud_lwp'][:].data
        iwp = dat.variables['cloud_iwp'][:].data

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
    y  = np.zeros((ns,nlev*2),dtype=np.float32)
    # inputs: one input vector consists of the atmospheric profile (T,p,gases),
    # plus mu0 (1) plus surface albedo (1)...these need to be flattened and
    # stacked on top of each other
    x_gasopt = x_gasopt.reshape((ns,nlay,nx))

    if add_coldry:
        pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
        pres = np.reshape(pres,(ns,nlev))
        coldry = get_col_dry(vmr_h2o,pres)
        coldry = coldry.reshape(ns,nlay,1)
        x_gasopt = np.concatenate((x_gasopt, coldry),axis=2)
        nx = nx + 1
    x_gasopt = np.reshape(x_gasopt,(nexp,ncol,nlay*nx)) # to profiles
    nx_gasopt = nlay*nx
    # nx_rte = ngpt + 1 # inc flux + albedo
    nx_rte = 1 + 1 # solar angle + albedo
    nx_clouds = 0
    if clouds: nx_clouds = 2*nlay
    x  = np.zeros((ns,(nx_gasopt + nx_rte + nx_clouds)),dtype=np.float32)
    # need to reshape mu0 and sfc alb so they are also 3D
    mu0     = np.reshape(mu0,(nexp,ncol,1))
    sfc_alb = np.reshape(sfc_alb,(nexp,ncol,1))
    print("nx gasopt {} rte {} clouds {}  nlay {}".format(nx, nx_rte, 2, nlay))
    i = 0
    if clouds:
        for iexp in range(nexp):
           for icol in range(ncol):
               y[i,:] = np.concatenate((y0[iexp,icol,:], y1[iexp,icol,:]))
               x[i,:] = np.concatenate((x_gasopt[iexp,icol,:], lwp[iexp,icol,:],
                    iwp[iexp,icol,:], mu0[iexp,icol],  sfc_alb[iexp,icol]))
               i = i + 1
    else:
        for iexp in range(nexp):
           for icol in range(ncol):
               y[i,:] = np.concatenate((y0[iexp,icol,:], y1[iexp,icol,:]))
               x[i,:] = np.concatenate((x_gasopt[iexp,icol,:], mu0[iexp,icol], sfc_alb[iexp,icol]))
               i = i + 1
           
    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))

    if not return_pressures:
        dat.close()
        return x,y
    else:
        pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
        pres = np.reshape(pres,(ns,nlev))
        dat.close()
        return x,y,pres
    
def load_radscheme_rnn(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False):
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    # argument scale_p_h2o_o3 determines whether specific gas optics inputs
    # (pressure, H2O and O3 )  are power-scaled similarly to Ukkonen 2020 paper
    # for a more normal distribution
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gas = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    (nexp,ncol,nlay,nx) = x_gas.shape
    nlev = nlay+1
    ns = nexp*ncol # number of samples (profiles)
    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,:,1] = np.log(x_gas[:,:,:,1])
        vmr_h2o = x_gas[:,:,:,2].reshape(ns,nlay)
        x_gas[:,:,:,2] = x_gas[:,:,:,2]**(1.0/4) 
        x_gas[:,:,:,3] = x_gas[:,:,:,3]**(1.0/4)
    
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
    
    lwp = dat.variables['cloud_lwp'][:].data
    iwp = dat.variables['cloud_iwp'][:].data

    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    rsu = dat.variables['rsu'][:]
    rsd = dat.variables['rsd'][:]
        
    if np.size(rsu.shape) != 3:
        sys.exit("Invalid array shapes, RTE output should have 3 dimensions")
    
    # Reshape to profiles...
    x_gas   = np.reshape(x_gas,(ns,nlay,nx)) 
    lwp     = np.reshape(lwp,  (ns,nlay,1))    
    iwp     = np.reshape(iwp,  (ns,nlay,1))
    rsu     = np.reshape(rsu,  (ns,nlev))
    rsd     = np.reshape(rsd,  (ns,nlev))
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    rsd0    = rsd[:,0]
    rsd     = rsd / np.repeat(rsd0.reshape(-1,1), nlev, axis=1)
    # remove rsd0 from array
    rsd     = rsd[:,1:]
    # extract and remove upwelling flux at surface, this will be computed 
    # explicitly, resulting in NN outputs with consistent dimensions to input (nlay)
    rsu0    = rsu[:,-1]
    rsu     = rsu[:,0:-1]
    rsu     = rsu / np.repeat(rsd0.reshape(-1,1), nlay, axis=1)

    rsu     = rsu.reshape((ns,nlay,1))
    rsd     = rsd.reshape((ns,nlay,1))

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..
    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)
    
    # Concatenate inputs and outputs...
    x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = np.concatenate((rsd,rsu),axis=2)

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    pres = dat.variables['pres_level'][:,:,:].data       # (nexp,ncol, nlev)
    pres = np.reshape(pres,(ns,nlev))
    
    if return_coldry:
        coldry = get_col_dry(vmr_h2o,pres)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres,coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,pres
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw

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

def load_rte_sw(fname):
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
    # 
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol,ngpt)
    # sfc_alb = sfc_alb[:,:,0] # (nexp,ncol)
    # plus cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # ..and incoming flux times mu0
    # Incoming flux does not vary by column, but does have a spectral dependence
    # SINCE THE NN MODEL TREATS ALL G-POINTS EQUALLY, the spectrally 
    # variant incoming flux needs to be an input!
    inc_flux = dat.variables['toa_flux'][:].data # (nexp,ncol, ngpt)

    y0 = dat.variables['rsu_gpt'][:] # (nexp,ncol,nlev,ngpt)
    y1 = dat.variables['rsd_gpt'][:]
    
    nlev = nlay+1
    # Permute from (nexp,ncol,nlev,ngpt) to (nexp,ncol,ngpt,nlev)
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

    y0 = np.reshape(y0,(nexp*ncol*ngpt,nlev))
    y1 = np.reshape(y1,(nexp*ncol*ngpt,nlev))
    y = np.hstack((y0,y1))
    
    del y0,y1
        
    # inputs: one input vector consists of vertical profiles of tau+ssa+g,
    # plus mu0, inc flux and surface albedo..these variables all need to be flattened and
    # stacked on top of each other
    x = np.zeros((ns,(3*nlay + 1 + 1 + 1)),dtype=np.float32)
    # need to ensure mu0, inc flux and albedo are provided at every column and gpt
    mu0     = np.reshape(mu0,(nexp,ncol,1))
    mu0     = np.repeat(mu0,ngpt,axis=2)
    mu0     = np.reshape(mu0,(nexp*ncol*ngpt,1))
    sfc_alb = np.reshape(sfc_alb,(nexp*ncol*ngpt,1))
    inc_flux = np.reshape(inc_flux,(nexp*ncol*ngpt,1))
    # downward radiance is inc flux times mu0
    inc_flux = mu0 * inc_flux

    tau = np.reshape(tau,(ns,nlay))
    ssa = np.reshape(ssa,(ns,nlay))
    g = np.reshape(g,(ns,nlay))
    
    stack_x_vector(ns,nlay,x,tau,ssa,g,mu0,inc_flux, sfc_alb)    
    
    del tau,ssa,g

    print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    # broadband flux    
    y0_bb = dat.variables['rsu'][:]; y1_bb = dat.variables['rsd'][:]
    y0_bb = np.reshape(y0_bb,(nexp*ncol,nlev)); y1_bb = np.reshape(y1_bb,(nexp*ncol,nlev))
    y_bb = np.hstack((y0_bb,y1_bb))

    return x,y,y_bb

def load_reftrans(fname,half_clouds=False):
    # Load data for training an emulator for one component of the 
    # RADIATIVE TRANSFER  SOLVER (RTE) - reflectance-transmittance computations
    # Inputs are layer-wise optical properties (tau, ssa, g) + solar angle mu0,
    # and outputs are layer-wise rdif,tdif,rdir,tdir (diffuse and direct 
    # reflectance and transmittance)
    
    dat = Dataset(fname)
    
    nexp = dat.dimensions['expt'].size 
    ncol = dat.dimensions['site'].size 
    nlay = dat.dimensions['layer'].size
    ntot = nexp*ncol*nlay
    
    if half_clouds:
        # Sample the data so that half of the samples are from
        # cloudy layers, by selecting all cloudy layers and an equal amount
        # of random non-cloudy layers?
        cloudfrac = dat.variables['cloud_fraction'][:]
        inds_clouds = cloudfrac > 0.0
        ncloud = inds_clouds.sum()
        inds_clouds = np.reshape(inds_clouds,(ntot))
        inds_noclouds = ~inds_clouds
        # To integer index array
        inds_nocloud = np.where(inds_noclouds)[0]
        inds_cloud   = np.where(inds_clouds)[0]
        # Get indices of random non-cloudy samples
        inds_cloudrnd = np.sort(np.random.choice(inds_nocloud,ncloud,replace=False))
        # Add these to the cloudy indices
        inds_sel = np.concatenate((inds_cloud,inds_cloudrnd))
        inds_sel.sort()
        # into a tuple of coordinate arrays (3D indices)
        ii,jj,kk = np.unravel_index(inds_sel,cloudfrac.shape)
    else:
        # All indices
        ii,jj,kk = np.unravel_index(np.arange(ntot),(nexp,ncol,nlay))
    
    # Inputs
    # nexp, ncol, nlay, ngpt
    if 'tau_sw' in dat.variables:
        tau = dat.variables['tau_sw'][:].data[ii,jj,kk,:]
    elif ('tau_sw_gas' in dat.variables):
        tau = dat.variables['tau_sw_gas'][:].data[ii,jj,kk,:] 
    else:
        print("couldn't find variable tau_sw or tau_sw_gas in netCDF file")
        
    # (nexp,ncol,nlay,ngpt) = tau.shape
    (nsel,ngpt) = tau.shape 
    
    if 'ssa_sw' in dat.variables:
        ssa = dat.variables['ssa_sw'][:].data[ii,jj,kk,:]
    elif ('ssa_sw_gas' in dat.variables):
        ssa = dat.variables['ssa_sw_gas'][:].data[ii,jj,kk,:] 
    else:
        print("couldn't find variable ssa_sw or ssa_sw_gas in netCDF file")    
    if 'g_sw' in dat.variables:
        g = dat.variables['g_sw'][:].data[ii,jj,kk,:]
    else:
        print("couldn't find variable g_sw in netCDF file, filling with zeroes")    
        # g = np.zeros((nexp,ncol,nlay,ngpt),dtype=np.float32)
        g = np.zeros((nsel,ngpt),dtype=np.float32)
    if 'mu0' in dat.variables:
        mu0 = dat.variables['mu0'][:].data[ii,jj]  # (nexp,ncol)
    else:
        print("couldn't find mu0 in netCDF file")
        
        
    # Outputs
    # Like inputs (nexp, ncol, nlay, ngpt), but flattened to (nsel,ngpt) 
    # where nsel are the selected exp, col and layer indices, flattened into 1D
    rdif = dat.variables['rdif'][:].data[ii,jj,kk,:]  # nexp, ncol, nlay, ngpt
    tdif = dat.variables['tdif'][:].data[ii,jj,kk,:]
    rdir = dat.variables['rdir'][:].data[ii,jj,kk,:]
    tdir = dat.variables['tdir'][:].data[ii,jj,kk,:]

    # (nexp,ncol,nlay,ngpt) = rdif.shape

    # Reshape to 2D data matrix 
    # ns = nexp*ncol*nlay*ngpt # number of samples 
    ns = nsel*ngpt
    rdif = np.reshape(rdif,(ns,1))
    tdif = np.reshape(tdif,(ns,1))
    rdir = np.reshape(rdir,(ns,1))
    tdir = np.reshape(tdir,(ns,1))

    tau = np.reshape(tau,(ns,1))
    ssa = np.reshape(ssa,(ns,1))
    g   = np.reshape(g,  (ns,1))
    
    # mu0 = np.reshape(mu0,(nexp,ncol,1,1))
    # mu0 = np.repeat(mu0,nlay,axis=2)
    # mu0 = np.repeat(mu0,ngpt,axis=3)

    mu0 = np.reshape(mu0,(nsel,1))
    mu0 = np.repeat(mu0,ngpt,axis=1)
    mu0 = np.reshape(mu0,(ns,1))

    x = np.hstack((tau,ssa,g,mu0))
    y = np.hstack((rdif,tdif,rdir,tdir))
    
    print( "{:e} samples were extracted from this dataset".format(ns))
    if half_clouds: print("50% of these are from cloudy layers")
    return x,y


@njit(parallel=True)
def reftrans_gammas(tau,w0,g,mu0):
    ns = tau.shape[0]

    gamma1 = np.zeros(ns,dtype=np.float64)
    gamma2 = np.zeros(ns,dtype=np.float64)
    gamma3 = np.zeros(ns,dtype=np.float64)

    for i in prange(ns):
        # Zdunkowski Practical Improved Flux Method "PIFM"
        #  (Zdunkowski et al., 1980;  Contributions to Atmowpheric Physics 53, 147-66)
        #
        gamma1[i]= (8. - w0[i] * (5. + 3. * g[i])) * .25
        gamma2[i]=  3. *(w0[i] * (1. -      g[i])) * .25
        gamma3[i]= (2. - 3. * mu0[i] *      g[i] ) * .25
   
    return gamma1,gamma2,gamma3

@njit(parallel=True)
def reftrans(tau,w0,g,mu0):
    ns = tau.shape[0]

    Tdir = np.zeros(ns,dtype=np.float64)
    Rdir = np.zeros(ns,dtype=np.float64)
    Tdif = np.zeros(ns,dtype=np.float64)
    Rdif = np.zeros(ns,dtype=np.float64)

    for i in prange(ns):
        mu0_inv = 1./mu0[i]
        Tnoscat = np.exp(-tau[i]*mu0_inv)
        epsilon = np.finfo(np.float64).eps

        # Zdunkowski Practical Improved Flux Method "PIFM"
        #  (Zdunkowski et al., 1980;  Contributions to Atmowpheric Physics 53, 147-66)
        #
        gamma1= (8. - w0[i] * (5. + 3. * g[i])) * .25
        gamma2=  3. *(w0[i] * (1. -         g[i])) * .25
        gamma3= (2. - 3. * mu0[i] *              g[i] ) * .25
        gamma4=  1. - gamma3
    
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3           # Eq. 16
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4           # Eq. 17
    
        k = np.sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2),  1e-12))

        exp_minusktau = np.exp(-tau[i]*k)
        
        #
        # Diffuse reflection and transmission
        #
        #$OMP SIMD
        exp_minus2ktau  = exp_minusktau * exp_minusktau

        # Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term = 1. / (k * (1. + exp_minus2ktau) + gamma1 * (1. - exp_minus2ktau) )

        # Equation 25
        Rdif[i] = RT_term * gamma2 * (1. - exp_minus2ktau)

        # Equation 26
        Tdif[i] = RT_term * 2. * k * exp_minusktau

        k_mu     = k * mu0[i]
        k_mu2    = k_mu*k_mu
        k_gamma3 = k * gamma3
        k_gamma4 = k * gamma4
        #
        # Equation 14, multiplying top and bottom by exp_fast(-k*tau)
        #   and rearranging to avoid div by 0.    
        x = 1 - k_mu2
        if np.abs(x) >= epsilon:
            RT_term = (w0[i] * RT_term) / x
        else:
            RT_term = (w0[i] * RT_term) / epsilon  


        Rdir[i] = RT_term  *                          \
            (   (1.0 - k_mu) * (alpha2 + k_gamma3) -  \
                (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau - \
             2.0 * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau  * Tnoscat)

        Tdir[i] = -RT_term *                                                \
            ((1.0 + k_mu) * (alpha1 + k_gamma4)                  * Tnoscat  \
            - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat \
            - 2.0 * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau )
                
            
                
    return Rdif, Tdif, Rdir, Tdir

def gen_synthetic_inp_outp_reftrans(ns, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0):
    print("Generating {:e} hypercube samples".format(ns))

    from skopt.sampler import Halton
    halton = Halton()

    if minmax_g == None:
        dims = [    (minmax_tau[0], minmax_tau[1]), 
                    (minmax_ssa[0], minmax_ssa[1]),
                    (minmax_mu0[0], minmax_mu0[1])]
        vals = np.array(halton.generate(dims, n_samples = ns))
        tau = vals[:,0]
        ssa = vals[:,1]
        mu0 = vals[:,2]
        
        g = np.zeros(ns)
    else:
        dims = [    (minmax_tau[0], minmax_tau[1]), 
                    (minmax_ssa[0], minmax_ssa[1]),
                    (minmax_g[0],   minmax_g[1]),
                    (minmax_mu0[0], minmax_mu0[1])]
        vals = np.array(halton.generate(dims, n_samples = ns))
        tau = vals[:,0]
        ssa = vals[:,1]
        g   = vals[:,2]
        mu0 = vals[:,3]
        
    str1="Doing reflectance-transmittance computations for {:e}".format(ns) \
        + " samples"
    print(str1)
    Rdif, Tdif, Rdir, Tdir = reftrans(tau,ssa,g,mu0)

    tau = np.reshape(tau,(ns,1))
    ssa = np.reshape(ssa,(ns,1))
    g   = np.reshape(g,  (ns,1))
    mu0 = np.reshape(mu0,(ns,1))
    
    Rdif = np.reshape(Rdif,(ns,1))
    Tdif = np.reshape(Tdif,(ns,1))
    Rdir = np.reshape(Rdir,(ns,1))
    Tdir = np.reshape(Tdir,(ns,1))
    
    x = np.hstack((tau,ssa,g,mu0))
    y = np.hstack((Rdif,Tdif,Rdir,Tdir))
    
    x = np.float32(x)
    y = np.float32(y)
    
        
    return x,y


@njit(parallel=True)
def stack_x_vector(ns,nlay,x,tau,ssa,g,mu0,inc_flux,sfc_alb):
    nlay2 = 2*nlay
    nlay3 = 3*nlay
    for i in prange(ns):
        x[i,0:nlay]         = tau[i,:]
        x[i,nlay:nlay2]     = ssa[i,:]
        x[i,nlay2:nlay3]    = g[i,:]
        x[i,nlay3:nlay3+1]  = mu0[i]
        x[i,nlay3+1:nlay3+2]= inc_flux[i]
        x[i,nlay3+2:nlay3+3]=sfc_alb[i]

def preproc_tau_to_crossection(tau, col_dry):
    y = np.zeros(tau.shape,dtype=np.float32)
    for igpt in range(tau.shape[1]):
        y[:,igpt]  = tau[:,igpt] / col_dry
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
            from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
            from sklearn.preprocessing import MinMaxScaler, StandardScaler

            scaler = MinMaxScaler()  
            scaler.fit(x_scaled)
            x_scaled = scaler.transform(x_scaled) 
            return x_scaled, scaler.data_min_, scaler.data_min_

        else:
            (xmin,xmax) = xcoeffs
            for i in range(x.shape[1]):
                x_scaled[:,i] =  (x_scaled[:,i] - xmin[i]) / (xmax[i] - xmin[i] )
            return x_scaled

# A wrapping function to automate things further for RRTMGP preprocessing        
def scale_gasopt(x_raw, y_raw, col_dry, scale_inputs=False, scale_outputs=False, 
                 nfac=1, y_mean=0, y_sigma=0, xcoeffs=None):

    if scale_inputs:
        if xcoeffs is None:
            x,xmin,xmax = preproc_minmax_inputs_rrtmgp(x_raw)
        else:
            x = preproc_minmax_inputs_rrtmgp(x_raw,xcoeffs )
    else:
        x = x_raw
        
    if scale_outputs:
        # Standardization coefficients loaded from file
#        y_mean = ymeans_sw_abs; y_sigma = ysigma_sw_abs
        # Set power scaling coefficient (y == y**(1/nfac))
        # nfac = 8 
        
        # Scale by layer number of molecules to obtain absorption cross section
        y   = preproc_tau_to_crossection(y_raw, col_dry)
        # Scale using power-scaling followed by standard-scaling
        y   = preproc_pow_standardization(y, nfac, y_mean, y_sigma)
    else:
        y = y_raw
        
    if xcoeffs is None:
        return x, y, xmin, xmax
    else: return x,y
    