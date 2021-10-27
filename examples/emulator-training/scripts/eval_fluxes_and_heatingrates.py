#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:45:11 2021

@author: peter
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
import cartopy.crs as ccrs

# import xarray as xr
# import cartopy.crs as ccrs
# import num2tex



R = 287 # J kg-1 K-1
g = 9.81 # m s-2
cp = 1004 # J K-1  kg-1


def toa_forcing_lbl(fluxup_ref,fluxup_nn,fluxup_lbl,ind0=0, ind=5):
    
    fluxup_ref1 = fluxup_ref[ind0,:,:] - fluxup_ref[ind,:,:]
    fluxup_nn1 = fluxup_nn[ind0,:,:] - fluxup_nn[ind,:,:]
    fluxup_lbl1 = fluxup_lbl[ind0,:,:] - fluxup_lbl[ind,:,:]
    
    fluxup_ref1_TOA = np.sum(fluxup_ref1[:,0])
    fluxup_nn1_TOA = np.sum(fluxup_nn1[:,0])
    fluxup_lbl1_TOA = np.sum(fluxup_lbl1[:,0])
    
    
    print( "TOA forcing for LineByLine: {:0.3f} W/m2 (GLOBAL) ".format(fluxup_lbl1_TOA))
    print( "TOA forcing for RRTMGP-REF: {:0.3f} W/m2 (GLOBAL) ".format(fluxup_ref1_TOA))
    print( "TOA forcing for RRTMGP-NN : {:0.3f} W/m2 (GLOBAL) ".format(fluxup_nn1_TOA))
    return fluxup_ref1_TOA,fluxup_nn1_TOA,fluxup_lbl1_TOA



def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def rmse_prof(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=0))

def rmse_tot(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions,targets,ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff),axis=ax)

def rmsee(diff,ax):
    return np.sqrt(((diff) ** 2).mean(axis=ax))


def bias(var1, var2,ax):
    return np.mean(var1-var2,axis=ax)

# def calc_heatingrate(fluxup,fluxdn,p):
#     dF = np.gradient(F,axis=1)
#     dp = np.gradient(p,axis=1)
#     dF = F[:,1:] - F[:,0:-1] 
#     dp = p[:,1:] - p[:,0:-1] 
    
#     dFdp = dF/dp
    
#     dTdt = (g/cp)*(dFdp) # K / s
#     dTdt_day = (24*3600)*dTdt
#     return dTdt_day

def calc_heatingrate(F, p):
    dF = F[:,1:] - F[:,0:-1] 
    dp = p[:,1:] - p[:,0:-1] 
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day



# Solar zenith angles above 90 degrees were replaced with 0 in the computations
# here those nighttime columns should probably be excluded
exclude_night_cols = True

rootdir = "../fluxes/"

var1 = 'rsd'
var2 = 'rsu'

fname_ref       = rootdir+'CAMS_2015_rsud_REFERENCE.nc'
fname_reftrans  = rootdir+'CAMS_2015_rsud_REFTRANS.nc'
fname_rrtmgp    = rootdir+'CAMS_2015_rsud_RRTMGP.nc'
fname_rrtmgp_old = rootdir+'CAMS_2015_rsud_RRTMGP_2020.nc'
fname_radscheme = rootdir+'CAMS_2015_rsud_RADSCHEME.nc'
fname_radscheme = rootdir+'tmp.nc'
fname_radscheme_r = rootdir+'CAMS_2015_rsud_RADSCHEME_RNN.nc'


dat_ref         =  Dataset(fname_ref)
dat_reftrans    =  Dataset(fname_reftrans)
dat_rrtmgp      =  Dataset(fname_rrtmgp)
dat_rrtmgp_old  =  Dataset(fname_rrtmgp_old)
dat_radscheme   =  Dataset(fname_radscheme)
dat_radscheme_r =  Dataset(fname_radscheme_r)

p   = dat_ref.variables['pres_level'][:,:,:].data
lon = np.rad2deg(dat_ref.variables['clon'][:].data)
lat = np.rad2deg(dat_ref.variables['clat'][:].data)
timedat = dat_ref.variables['time'][:]

ntime = p.shape[0]
nsite = p.shape[1]
nlev = p.shape[2]

# shape = (ntime*nsite,nlev)
shape = (ntime,nsite,nlev)

p = p.reshape(shape)

fluxdn_ref          = dat_ref.variables[var1][:,:,:].data.reshape(shape)
fluxup_ref          = dat_ref.variables[var2][:,:,:].data.reshape(shape)
fluxdn_reftrans     = dat_reftrans.variables[var1][:,:,:].data.reshape(shape)
fluxup_reftrans     = dat_reftrans.variables[var2][:,:,:].data.reshape(shape)
fluxdn_rrtmgp_old   = dat_rrtmgp_old.variables[var1][:,:,:].data.reshape(shape)
fluxup_rrtmgp_old   = dat_rrtmgp_old.variables[var2][:,:,:].data.reshape(shape)
fluxdn_rrtmgp       = dat_rrtmgp.variables[var1][:,:,:].data.reshape(shape)
fluxup_rrtmgp       = dat_rrtmgp.variables[var2][:,:,:].data.reshape(shape)
fluxdn_radscheme    = dat_radscheme.variables[var1][:,:,:].data.reshape(shape)
fluxup_radscheme    = dat_radscheme.variables[var2][:,:,:].data.reshape(shape)
fluxdn_radscheme_r  = dat_radscheme_r.variables[var1][:,:,:].data.reshape(shape)
fluxup_radscheme_r  = dat_radscheme_r.variables[var2][:,:,:].data.reshape(shape)


cloudfrac = dat_ref.variables['cloud_fraction'][:]
sza = dat_ref.variables['solar_zenith_angle'][:]

nightcols = (sza > 89.99)
# nightcols_flat = nightcols.flatten()

if exclude_night_cols:
    for var in [fluxdn_ref, fluxup_ref, fluxdn_reftrans, fluxup_reftrans,
            fluxdn_rrtmgp_old, fluxup_rrtmgp_old, fluxdn_rrtmgp, fluxup_rrtmgp,
            fluxdn_radscheme, fluxup_radscheme, fluxdn_radscheme_r, fluxup_radscheme_r,
            ]:
        var[nightcols,:] = 0.0

rsu_toa_ref         = fluxup_ref[:,:,0]
rsu_toa_reftrans    = fluxup_reftrans[:,:,0]
rsu_toa_rrtmgp      = fluxup_rrtmgp[:,:,0]
rsu_toa_rrtmgp_old  = fluxup_rrtmgp_old[:,:,0]
rsu_toa_radscheme   = fluxup_radscheme[:,:,0]
rsu_toa_radscheme_r = fluxup_radscheme_r[:,:,0]

rsd_toa_ref         = fluxdn_ref[:,:,0]
rsd_sfc_ref         = fluxdn_ref[:,:,-1]
rsd_sfc_reftrans    = fluxdn_reftrans[:,:,-1]
rsd_sfc_rrtmgp      = fluxdn_rrtmgp[:,:,-1]
rsd_sfc_rrtmgp_old  = fluxdn_rrtmgp_old[:,:,-1]
rsd_sfc_radscheme   = fluxdn_radscheme[:,:,-1]
rsd_sfc_radscheme_r = fluxdn_radscheme_r[:,:,-1]


shape = (ntime*nsite,nlev)

for var in [p, fluxdn_ref, fluxup_ref, fluxdn_reftrans, fluxup_reftrans,
        fluxdn_rrtmgp_old, fluxup_rrtmgp_old, fluxdn_rrtmgp, fluxup_rrtmgp,
        fluxdn_radscheme, fluxup_radscheme, fluxdn_radscheme_r, fluxup_radscheme_r]:
    # var = var.reshape(shape)   
    var.shape = shape
    

fluxnet_ref         = fluxdn_ref - fluxup_ref
fluxnet_reftrans    = fluxdn_reftrans - fluxup_reftrans
fluxnet_rrtmgp_old  = fluxdn_rrtmgp_old - fluxup_rrtmgp_old
fluxnet_rrtmgp      = fluxdn_rrtmgp - fluxup_rrtmgp
fluxnet_radscheme    = fluxdn_radscheme - fluxup_radscheme 
fluxnet_radscheme_r  = fluxdn_radscheme_r - fluxup_radscheme_r
   
dTdt_ref        = calc_heatingrate(fluxnet_ref,p)
dTdt_reftrans   = calc_heatingrate(fluxnet_reftrans,p)
dTdt_rrtmgp_old = calc_heatingrate(fluxnet_rrtmgp_old,p)
dTdt_rrtmgp     = calc_heatingrate(fluxnet_rrtmgp,p)
dTdt_radscheme  = calc_heatingrate(fluxnet_radscheme,p)
dTdt_radscheme_r = calc_heatingrate(fluxnet_radscheme_r,p)

p_lay = 0.5 * (p[:,1:] + p[:,0:-1])


dat_ref.close()
dat_reftrans.close()
dat_rrtmgp.close()
dat_rrtmgp_old.close()
dat_radscheme.close()
dat_radscheme_r.close()



# FLUX AND HEATING RATE ERRORS: feedforward NN radscheme emulators, validation data

def plot_evalfig1():
    
    from ml_loaddata import preproc_standardization_reverse
    from ml_eval_funcs import heatingrate_stats, plot_flux_and_hr_error, mae

    y_pred_noscale          = np.load('evalfig1/noscale.npy')
    # y_pred_stdscale         = np.load('evalfig1/stdscale.npy')
    y_pred_toascale         = np.load('evalfig1/toascale.npy')
    y_pred_toascale_hrloss  = np.load('evalfig1/toascale_hrloss.npy')
    y_fluxes                = np.load('evalfig1/fluxes.npy')
    pres_val                = np.load('evalfig1/pres_val.npy')

    # HR model
    y_pred_hr_scaled        = np.load('evalfig1/hr.npy')  
    y_hr_scaled             = np.load('evalfig1/hr_ref.npy')
    ny = y_hr_scaled.shape[1]

    # From the HR predicting model outputs, get the unscaled HR and fluxes
    fpath_ycoeffs = "../../../neural/data/nn_radscheme_hr_std_scaling_coeffs.txt"
    ycoeffs = np.loadtxt(fpath_ycoeffs, delimiter=',')
    y_mean = np.float32(ycoeffs[0:ny])
    y_sigma = np.float32(ycoeffs[ny]); y_sigma = np.repeat(y_sigma,ny)

    y_pred_hr  = preproc_standardization_reverse(y_pred_hr_scaled, y_mean,y_sigma)
    y_hr    = preproc_standardization_reverse(y_hr_scaled, y_mean, y_sigma)

    rsd_sfc_pred = y_pred_hr[:,-1]
    rsu_sfc_pred = y_pred_hr[:,-3]; rsu_toa_pred = y_pred_hr[:,-4]
    rsd_sfc_true = y_hr[:,-1]
    rsu_sfc_true  = y_hr[:,-3]; rsu_toa_true  = y_hr[:,-4]
    # reduce to just the heating rate profiles, without fluxes
    y_pred_hr = y_pred_hr[:,0:-4]
    y_hr   = y_hr[:,0:-4]
        
    mae_rsu_toa = mae(rsu_toa_true, rsu_toa_pred); mae_rsu_sfc = mae(rsu_sfc_true, rsu_sfc_pred)
    mae_rsd_sfc = mae(rsd_sfc_true, rsd_sfc_pred)


    # PLOT FLUX AND HEATING RATE ERRORS FOR DIFFERENT RADSCHEME EMULATORS
    
    # start index of y axis used for evaluation: 0 is the top of atmosphere (0.01 Pa)
    ind_p = 0
    # ind_p = 21 # 50 hPa
    
    xmax = [19,19,10]
    fig, axes = plt.subplots(nrows=4, ncols=3)
    plot_flux_and_hr_error(y_fluxes, y_pred_noscale, pres_val, axes[0,:], xlabel=False, xmax=xmax, ind_p=ind_p)
    plot_flux_and_hr_error(y_fluxes, y_pred_toascale, pres_val, axes[1,:], xlabel=False, xmax=xmax, ind_p=ind_p)
    # plot_flux_and_hr_error(y_fluxes, y_pred_toascale_hrloss, pres_val,axes[2,:], xmax=xmax, ind_p = ind_p)
    plot_flux_and_hr_error(y_fluxes, y_pred_toascale_hrloss, pres_val,axes[2,:], xlabel=False, xmax=xmax, ind_p = ind_p)

    # Last row: model predicting heating rates
    str1,str2, str3, str4 = heatingrate_stats(y_hr[:,ind_p:], y_pred_hr[:,ind_p:])

    str_rsu_toa = 'MAE, TOA: {:0.2f} ({:0.1f}%)'.format(mae_rsu_toa, 
                                100 * np.abs(mae_rsu_toa / rsu_toa_true.mean()))
    str_rsu_sfc = 'MAE, sfc: {:0.2f} ({:0.1f}%)'.format(mae_rsu_sfc, 
                                100 * np.abs(mae_rsu_sfc / rsu_sfc_true.mean()))
    str_rsd_sfc = 'MAE, sfc: {:0.2f} ({:0.1f}%)'.format(mae_rsd_sfc, 
                                100 * np.abs(mae_rsd_sfc / rsd_sfc_true.mean()))

    ts = 10
    xb = 0.45
    xyc1 = (xb, 0.87); xyc2 = (xb, 0.79)
    xyc3 = (xb, 0.71); xyc4 = (xb, 0.63)
    xycc = 'axes fraction'
        
    pres_lay = 0.5 * (pres_val[:,1:] + pres_val[:,0:-1])

    axes[3,2].plot(mae(y_pred_hr[:,ind_p:], y_hr[:,ind_p:]),  0.01*pres_lay[:,ind_p:].mean(axis=0))
    # axes[3,0].set_ylabel('Pressure (hPa)',fontsize=12)
    axes[3,2].set_xlim([0,xmax[2]]); axes[3,2].invert_yaxis();  axes[3,2].grid()
    axes[3,2].annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
    axes[3,2].annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
    axes[3,2].annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
    axes[3,2].annotate(str4, xy=xyc4, xycoords=xycc,size=ts)
        
    axes[3,0].annotate(str_rsd_sfc, xy=xyc2, xycoords=xycc,size=ts)
    axes[3,1].annotate(str_rsu_sfc, xy=xyc2, xycoords=xycc,size=ts)
    axes[3,1].annotate(str_rsu_toa, xy=xyc3, xycoords=xycc,size=ts)

    axes[3,0].set_xlabel('MAE in downwelling flux (W m$^{-2}$)',fontsize=12); 
    axes[3,1].set_xlabel('MAE in upwelling flux (W m$^{-2}$)',fontsize=12); 
    axes[3,2].set_xlabel('MAE in heating rate (K d$^{-1}$)',fontsize=12) 

    axes[3,0].xaxis.set_ticks([]); axes[3,0].yaxis.set_ticks([])
    axes[3,1].xaxis.set_ticks([]); axes[3,1].yaxis.set_ticks([])

    for axx in [axes[0,1], axes[0,2], axes[1,1], axes[1,2], axes[2,1], axes[2,2]]:
        axx.yaxis.set_ticklabels([])


    rowtitles = ['a) No scaling', 'b) Physical scaling', 'c) Physical scaling\n+ HR loss', 'd) HR prediction +\nstandard scaling']

    for axx, row in zip(axes[:,0], rowtitles):
        axx.annotate(row, xy=(0, 0), xytext=(-60, 60),
          xycoords='axes fraction', textcoords='offset points',
          size=14, ha='right', va='center', rotation=0)      

    for axx, row in zip(axes[0,:], ['Downwelling flux','Upwelling flux','Heating rate']):
        axx.annotate(row, xy=(0, 0), xytext=(120, 160),
          xycoords='axes fraction', textcoords='offset points',
          size=14, ha='center', va='center', rotation=0)
          


# Flux and heating rate errors: all models

def plot_evalfig2():
    
    l_mae = 'Mean abs. error'
    l_bias = 'Mean error'
    
    fluxes = [[fluxdn_radscheme, fluxup_radscheme], 
              [fluxdn_radscheme_r, fluxup_radscheme_r],
              [fluxdn_reftrans,  fluxup_reftrans],
              [fluxdn_rrtmgp,   fluxup_rrtmgp]]
        
    fluxes_ref = [fluxdn_ref, fluxup_ref]
    
    heatingrates = [dTdt_radscheme, dTdt_radscheme_r, dTdt_reftrans, dTdt_rrtmgp]
    
    xlim = [-10, 10]
    xlim2 = [-4,4]
    fs_cols = 14
    fs_ax   = 13
    ts = 9
    xb = 0.62
    xyc1 = (xb, 0.87)
    xyc2 = (xb, 0.79)
    xyc3 = (xb, 0.71)
    xyc4 = (xb, 0.63)
    xycc = 'axes fraction'
    ind_p = 0
    # ind_p = 4
    xlim2 = [-1.5,1.5]

    yy = 0.01*p[:,:].mean(axis=0)
    yy2 = 0.01*p_lay[:,:].mean(axis=0)
    
    yy = yy[ind_p:]; yy2 = yy2[ind_p:]
        
    ncols = 3 # metrics
    nrows = 4 # models
    fig, ax = plt.subplots(nrows,ncols)#, sharey=True)
    
    for i in range(nrows): # different models
        for j in range(2): # downwelling, then shortwelling
            var    = fluxes[i][j]; varref = fluxes_ref[j]
            var = var[:,ind_p:]; varref = varref[:,ind_p:]

            diff = var - varref
            
            p5  = np.percentile(diff, 5, axis=0); p95 = np.percentile(diff, 95, axis=0)
            xb  = np.mean(diff, axis=0)
            x = np.mean(np.abs(diff),axis=0)
            
            ax[i,j].plot(x, yy,'b', label=l_mae, ls='--')
            ax[i,j].plot(xb, yy,'b', label=l_bias)    
            ax[i,j].fill_betweenx(yy, p5, p95, color='b', alpha=.1)
            ax[i,j].set_xlim(xlim)
            
            if (i==1 and j==0):  ax[i,j].legend(fontsize=10,loc='center left')
            
            if (i==0): 
                if (j==0):
                    ax[i,j].set_title('Downwelling flux', fontsize=fs_cols)
                else:
                    ax[i,j].set_title('Upwelling flux', fontsize=fs_cols)
    
            if (j==0): ax[i,j].set_ylabel('Pressure (hPa)',fontsize=fs_ax)
            if (i==3): ax[i,j].set_xlabel('Flux (W m$^{-2}$)',fontsize=fs_ax); 
            
            str1= 'MAE : {:0.2f} ({:0.1f}%)'.format(x.mean(), 100*np.abs(x.mean()/varref.mean()))
            str2= 'Bias: {:0.2f} ({:0.1f}%)'.format(xb.mean(), 100*xb.mean()/varref.mean())
            str3= 'RMSE: {:0.2f}'.format(rmse_tot(var, varref))
            err =  np.corrcoef(var.flatten(),varref.flatten())[0,1]; err = err**2
            str4= 'R$^2$: {:0.5f}'.format(err)
    
            ax[i,j].annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
            ax[i,j].annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
            ax[i,j].annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
            ax[i,j].annotate(str4, xy=xyc4, xycoords=xycc,size=ts)
    
        # Heating rates
        
        j = 2
        var    = heatingrates[i]
        varref = dTdt_ref
        var = var[:,ind_p:]; varref = varref[:,ind_p:]

        diff = var - varref
        
        p5  = np.percentile(diff, 5, axis=0)
        p95 = np.percentile(diff, 95, axis=0)
        xb  = np.mean(diff, axis=0)
        x = np.mean(np.abs(diff),axis=0)
        
        ax[i,j].plot(x, yy2,'b', label=l_mae, ls='--')
        ax[i,j].plot(xb, yy2,'b', label=l_bias)    
        ax[i,j].fill_betweenx(yy2, p5, p95, color='b', alpha=.1)
        ax[i,j].set_xlim(xlim2)
        
        if (i==1 and j==1):  ax[i,j].legend(fontsize=10)
        
        if (i==0): 
            ax[i,j].set_title('Heating rate', fontsize=fs_cols)
            
        if (i==3): 
            ax[i,j].set_xlabel('Heating rate (K h$^{-1}$)',fontsize=fs_ax); 
    
        str1= 'MAE : {:0.2f} ({:0.1f}%)'.format(x.mean(), 100*np.abs(x.mean()/varref.mean()))
        str2= 'Bias: {:0.2f} ({:0.1f}%)'.format(xb.mean(), 100*xb.mean()/varref.mean())
        str3= 'RMSE: {:0.2f}'.format(rmse_tot(var, varref))
        err =  np.corrcoef(var.flatten(),varref.flatten())[0,1]; err = err**2
        str4= 'R$^2$: {:0.5f}'.format(err)
        
        ax[i,j].annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
        ax[i,j].annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
        ax[i,j].annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
        ax[i,j].annotate(str4, xy=xyc4, xycoords=xycc,size=ts)
    
    
    for i in range(nrows): # different models
        for j in range(ncols): 
                ax[i,j].invert_yaxis(); ax[i,j].grid()
    
    
    xpad = 70
    ypad = 0
    rowtitles = ['a) FNN-Radscheme', 'b) RNN-RadScheme', 'c) FNN-RefTrans', 'd) FNN-RRTMGP']
    for axx, row in zip(ax[:,0], rowtitles):
        axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad - xpad, 0 + ypad),
                    xycoords=axx.yaxis.label, textcoords='offset points',
                    size=14, ha='right', va='center')
    # for axx, row in zip(ax[:,0], rowtitles):
    #     axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad - xpad, 0 + ypad),
    #                 xycoords=axx.yaxis.label, textcoords='offset points',
    #                 size=14, ha='right', va='center', rotation=45)
    for axx in [ax[0,1], ax[0,2], ax[1,1], ax[1,2], ax[2,1], ax[2,2], ax[3,1], ax[3,2]]:
        axx.yaxis.set_ticklabels([])
        



def annotate_axis(ax,var,var_ref, xyc1, xyc2, xycc, ts):
    diff = np.abs(var-var_ref)
    varbias = (var-var_ref).mean() 
    str2= 'MAE : {:0.2f} ({:0.1f}%)'.format(diff.mean(), np.abs(100*diff.mean()/var_ref.mean()))
    str1= 'Bias: {:0.2f} ({:0.1f}%)'.format(varbias, 100*varbias/var_ref.mean())

    ax.annotate(str1, xy=xyc1, xycoords=xycc,size=ts,color='k')
    ax.annotate(str2, xy=xyc2, xycoords=xycc,size=ts,color='k')



# TOA upwelling flux errors
def plot_evalfig3():
    
    xycc = 'axes fraction'

    fs_bigtitle = 15
    z0 = rsu_toa_ref.mean(axis=0)
    func = bias
    # func = mae
    z1 = func(rsu_toa_ref,rsu_toa_radscheme,0)
    z2 = func(rsu_toa_ref,rsu_toa_radscheme_r,0)
    z3 = func(rsu_toa_ref,rsu_toa_reftrans,0)
    z4 = func(rsu_toa_ref,rsu_toa_rrtmgp,0)
    
    vmin = -4.0
    vmax = 4.0
    # bounds =  np.linspace(vmin,vmax,9)
    bounds =  np.linspace(vmin,vmax,17)
    
    ts2 = 11
    ts_rowtitles = 13
    xb = 1.02
    xyc1 = (xb, 0.67)
    xyc2 = (xb, 0.57)
    incl_absval = True
    coastcolor = 'gray'
    
    proj = ccrs.EqualEarth()
    x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T
    
    cmap = plt.get_cmap('RdBu')
    
    fig = plt.figure(figsize=(16,9))
    fig.suptitle('Upwelling flux at top-of-atmosphere', fontsize = fs_bigtitle)
    
    i = 1
    if incl_absval:
        rowtitles = ['(a) REF', '(b) REF - \nFNN-RadScheme',  '(c) REF - \nRNN-RadScheme',
                 '(d) REF - \nFNN-RefTrans', '(e) REF - \nFNN-RRTMGP']
        nrows = 5
        ax0 = plt.subplot(nrows, 1, i,projection=proj)
        cs0 = ax0.tricontourf(x, y, z0)
        ax0.coastlines(color='white')
        xvals_cbar = [1.15, -0.4, 0.08, 3.1]
        i = i +1
        
    else:
        rowtitles = ['(a) FNN-RadScheme',  '(b) RNN-RadScheme',
                 '(c) FNN-RefTrans', '(d) FNN-RRTMGP']
        xvals_cbar = [1.15, -0.6, 0.08, 2.5]
        nrows = 4
        
    xvals_cbar2 = [-0.25, -0.33, 1.5, 0.15]
    
    cax0= ax0.inset_axes([1.15 ,0.13, 0.08, 0.8], transform=ax0.transAxes)
    colorbar0 = fig.colorbar(cs0, cax=cax0); 
    cax0.set_xlabel('Flux (W m$^{-2}$)',fontsize=11,  labelpad=11)
    
    ax1 = plt.subplot(nrows, 1, i,projection=proj)
    cs1 = ax1.tricontourf(x, y, z1, levels=bounds, cmap=cmap)
    ax1.coastlines(color=coastcolor)
    annotate_axis(ax1,rsu_toa_radscheme,rsu_toa_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax2 = plt.subplot(nrows, 1, i,projection=proj)
    cs2 = ax2.tricontourf(x, y, z2, levels=bounds, cmap=cmap)
    ax2.coastlines(color=coastcolor)
    annotate_axis(ax2,rsu_toa_radscheme_r,rsu_toa_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax3 = plt.subplot(nrows, 1, i,projection=proj)
    cs3 = ax3.tricontourf(x, y, z3, levels=bounds, cmap=cmap)
    ax3.coastlines(color=coastcolor)
    annotate_axis(ax3,rsu_toa_reftrans,rsu_toa_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax4 = plt.subplot(nrows, 1, i, projection=proj)
    cs4 = ax4.tricontourf(x, y, z4, levels=bounds, cmap=cmap)
    ax4.coastlines(color=coastcolor)
    annotate_axis(ax4,rsu_toa_rrtmgp,rsu_toa_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    cax = ax4.inset_axes(xvals_cbar2, transform=ax4.transAxes)
    colorbar = fig.colorbar(cs4, cax=cax,orientation='horizontal') 
    cax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11, labelpad=11)
    
    xpad = 125
    ypad = 15
    
    if incl_absval:
        axrow = [ax0,ax1,ax2,ax3,ax4]
    else:
        axrow = [ax1,ax2,ax3,ax4]   
    
    
    for axx, row in zip(axrow, rowtitles):
        # axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad +xpad, 0 + ypad),
        #             xycoords=axx.yaxis.label, textcoords='offset points',
        #             size=14, ha='right', va='center', rotation=45)
        axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad +xpad, 0 + ypad),
          xycoords=axx.yaxis.label, textcoords='offset points',
          size=ts_rowtitles, ha='right', va='center', rotation=0)      
    
# SFC downwelling flux errors
def plot_evalfig4():
    
    xycc = 'axes fraction'

    fs_bigtitle = 15
    z0 = rsd_sfc_ref.mean(axis=0)
    func = bias
    # func = mae
    z1 = func(rsd_sfc_ref,rsd_sfc_radscheme,0)
    z2 = func(rsd_sfc_ref,rsd_sfc_radscheme_r,0)
    z3 = func(rsd_sfc_ref,rsd_sfc_reftrans,0)
    z4 = func(rsd_sfc_ref,rsd_sfc_rrtmgp,0)
    
    vmin = -6.0
    vmax = 6.0
    # bounds =  np.linspace(vmin,vmax,9)
    bounds =  np.linspace(vmin,vmax,17)
    
    ts2 = 11
    ts_rowtitles = 13
    xb = 1.02
    xyc1 = (xb, 0.67)
    xyc2 = (xb, 0.57)
    incl_absval = True
    coastcolor = 'gray'
    
    proj = ccrs.EqualEarth()
    x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T
    
    cmap = plt.get_cmap('RdBu')
    
    fig = plt.figure(figsize=(16,9))
    fig.suptitle('Downwelling flux at surface', fontsize = fs_bigtitle)
    
    i = 1
    if incl_absval:
        rowtitles = ['(a) REF', '(b) REF - \nFNN-RadScheme',  '(c) REF - \nRNN-RadScheme',
                 '(d) REF - \nFNN-RefTrans', '(e) REF - \nFNN-RRTMGP']
        nrows = 5
        ax0 = plt.subplot(nrows, 1, i,projection=proj)
        cs0 = ax0.tricontourf(x, y, z0)
        ax0.coastlines(color='white')
        xvals_cbar = [1.15, -0.4, 0.08, 3.1]
        i = i +1
        
    else:
        rowtitles = ['(a) FNN-RadScheme',  '(b) RNN-RadScheme',
                 '(c) FNN-RefTrans', '(d) FNN-RRTMGP']
        xvals_cbar = [1.15, -0.6, 0.08, 2.5]
        nrows = 4
        
    xvals_cbar2 = [-0.25, -0.33, 1.5, 0.15]
    
    cax0= ax0.inset_axes([1.15 ,0.13, 0.08, 0.8], transform=ax0.transAxes)
    colorbar0 = fig.colorbar(cs0, cax=cax0); 
    cax0.set_xlabel('Flux (W m$^{-2}$)',fontsize=11,  labelpad=11)
    
    ax1 = plt.subplot(nrows, 1, i,projection=proj)
    cs1 = ax1.tricontourf(x, y, z1, levels=bounds, cmap=cmap)
    ax1.coastlines(color=coastcolor)
    annotate_axis(ax1,rsd_sfc_radscheme,rsd_sfc_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax2 = plt.subplot(nrows, 1, i,projection=proj)
    cs2 = ax2.tricontourf(x, y, z2, levels=bounds, cmap=cmap)
    ax2.coastlines(color=coastcolor)
    annotate_axis(ax2,rsd_sfc_radscheme_r,rsd_sfc_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax3 = plt.subplot(nrows, 1, i,projection=proj)
    cs3 = ax3.tricontourf(x, y, z3, levels=bounds, cmap=cmap)
    ax3.coastlines(color=coastcolor)
    annotate_axis(ax3,rsd_sfc_reftrans,rsd_sfc_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    ax4 = plt.subplot(nrows, 1, i, projection=proj)
    cs4 = ax4.tricontourf(x, y, z4, levels=bounds, cmap=cmap)
    ax4.coastlines(color=coastcolor)
    annotate_axis(ax4,rsd_sfc_rrtmgp,rsd_sfc_ref, xyc1, xyc2, xycc, ts2)
    i = i +1
    
    cax = ax4.inset_axes(xvals_cbar2, transform=ax4.transAxes)
    colorbar = fig.colorbar(cs4, cax=cax,orientation='horizontal') 
    cax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11, labelpad=11)
    
    xpad = 125
    ypad = 15
    
    if incl_absval:
        axrow = [ax0,ax1,ax2,ax3,ax4]
    else:
        axrow = [ax1,ax2,ax3,ax4]   
    
    
    for axx, row in zip(axrow, rowtitles):
        # axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad +xpad, 0 + ypad),
        #             xycoords=axx.yaxis.label, textcoords='offset points',
        #             size=14, ha='right', va='center', rotation=45)
        axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad +xpad, 0 + ypad),
          xycoords=axx.yaxis.label, textcoords='offset points',
          size=ts_rowtitles, ha='right', va='center', rotation=0)      
    


plot_evalfig2()
f
plot_evalfig3()

plot_evalfig4()

# def plot_fig0():
#     import psyplot.project as psy    
    # from psyplot import open_dataset
    # from psy_maps.plotters import FieldPlotter
    
    # fname = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/fluxes/REF.nc"
    # ds = open_dataset(fname_ref)
    # maps = psy.plot.mapplot(ds, name='surface_temperature', projection='moll',
    #                         cmap='RdBu_r',  title='2-metre temperature',clabel="Temperature (K)")
    # clon = np.rad2deg(ds['clon'].data)
    # clat = np.rad2deg(ds['clat'].data)
    # z = ds['rsu'][0,0,:].data

# z0 = rsu_toa_ref.mean(axis=0)
# # z0 = rsd_sfc_ref.mean(axis=0)


# z0 = sza[7,:]
# # z0 = sza.mean(axis=0)
# # z0 = np.mean(cloudfrac.sum(axis=2),axis=0)

# x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T

# fig = plt.figure(figsize=(16,9))
# proj = ccrs.EqualEarth()

# proj = ccrs.EqualEarth()
# ax = plt.axes(projection=proj)
# cs = ax.tricontourf(x, y, z0)
# ax.coastlines(color='white')

# cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
# colorbar = fig.colorbar(cs, cax=cax)


