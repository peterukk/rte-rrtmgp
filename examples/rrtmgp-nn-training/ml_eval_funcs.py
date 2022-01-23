#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:01:44 2021

@author: puk
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# def rmse(y, y_pred):
#     return np.sqrt(np.mean(np.square(y - y_pred)))

# def mse(y, y_pred):
#     return np.mean(np.square(y - y_pred))

# def bias(y, y_pred):
#     return np.mean(y - y_pred)

def calc_heatingrates(fluxup,fluxdn, p):
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

def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def mse(predictions, targets,ax=0):
    return ((predictions - targets) ** 2).mean(axis=ax)

def mae(predictions,targets,ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff),axis=ax)

def plot_heatingrate_error(hr_true, hr_pred, pres):
    # errfunc = mae
    errfunc = rmse
    ind_p = 0
    hre_radscheme       = errfunc(hr_true[:,ind_p:], hr_pred[:,ind_p:])
    yy = 0.01*pres[:,:].mean(axis=0)
    figtitle = 'ERrror in shortwave heating rate'
    fig, ax = plt.subplots(1)
    ax.plot(hre_radscheme,  yy[ind_p:])
    ax.invert_yaxis(); ax.grid()
    ax.set_ylabel('Pressure (hPa)',fontsize=15)
    ax.set_xlabel('Heating rate (W m$^{-2}$)',fontsize=15); 
    str1,str2, str3, str4 = heatingrate_stats(hr_true, hr_pred)
    xb = 0.45; xycc = 'axes fraction'; ts = 10
    xyc1 = (xb, 0.87); xyc2 = (xb, 0.79)
    xyc3 = (xb, 0.71); xyc4 = (xb, 0.63)
    ax.annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
    ax.annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
    ax.annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
    ax.annotate(str4, xy=xyc4, xycoords=xycc,size=ts)
    
    
    fig.suptitle(figtitle, fontsize=16)
    
def heatingrate_stats(dTdt_true, dTdt_pred):
    bias_tot = np.mean(dTdt_pred.flatten()-dTdt_true.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_percent = 100 * np.abs(mae_tot / dTdt_true.mean())
    r2 =  np.corrcoef(dTdt_pred.flatten(),dTdt_true.flatten())[0,1]; r2 = r2**2
    
    str1= 'MAE : {:0.2f} ({:0.1f}%)'.format(mae_tot, mae_percent)
    str2= 'Bias: {:0.2f} ({:0.1f}%)'.format(bias_tot, np.abs(100*bias_tot/dTdt_true.mean()))
    str3= 'RMSE: {:0.2f}'.format(rmse_tot)
    str4= 'R$^2$: {:0.5f}'.format(r2)
    
    return str1, str2, str3, str4
    
def plot_flux_and_hr_error(y_true, y_pred, pres, ax=None, xlabel=True, xmax= None,ind_p=0):
    
    fluxup_true = y_true[:,0:61]; fluxdn_true = y_true[:,61:]
    fluxup_pred = y_pred[:,0:61]; fluxdn_pred = y_pred[:,61:]

    rsu_toa_true = fluxup_true[:,0]; rsu_sfc_true = fluxup_true[:,-1]
    rsd_sfc_true = fluxdn_true[:,-1]
    rsu_toa_pred = fluxup_pred[:,0]; rsu_sfc_pred = fluxup_pred[:,-1]
    rsd_sfc_pred = fluxdn_pred[:,-1]
    
    # start analysis from given ind_p (pressure index)
    fluxdn_true = fluxdn_true[:,ind_p:]; fluxup_true = fluxup_true[:,ind_p:]
    fluxdn_pred = fluxdn_pred[:,ind_p:]; fluxup_pred = fluxup_pred[:,ind_p:]
    pres = pres[:,ind_p:]
    
    ts = 10; ts_ax = 12
    xb = 0.45
    xyc1 = (xb, 0.87); xyc2 = (xb, 0.79)
    xyc3 = (xb, 0.71); xyc4 = (xb, 0.63)
    xycc = 'axes fraction'

    pres_lay = 0.5 * (pres[:,1:] + pres[:,0:-1])
    
    dTdt_true, fluxup_true, fluxdn_true = calc_heatingrates(fluxup_true, fluxdn_true, pres)
    dTdt_pred, fluxup_pred, fluxdn_pred = calc_heatingrates(fluxup_pred, fluxdn_pred, pres)
    
        
    str1,str2, str3, str4 = heatingrate_stats(dTdt_true, dTdt_pred)
        
    mae_rsu = mae(fluxup_true.flatten(), fluxup_pred.flatten())
    mae_rsd = mae(fluxdn_true.flatten(), fluxdn_pred.flatten())
    mae_rsu_p = 100 * np.abs(mae_rsu / fluxup_true.mean())
    mae_rsd_p = 100 * np.abs(mae_rsd / fluxdn_true.mean())
    
    mae_rsu_toa = mae(rsu_toa_true, rsu_toa_pred); mae_rsu_sfc = mae(rsu_sfc_true, rsu_sfc_pred)
    mae_rsd_sfc = mae(rsd_sfc_true, rsd_sfc_pred)

    str_rsu =  'MAE: {:0.2f} ({:0.1f}%)'.format(mae_rsu, mae_rsu_p)
    str_rsd =  'MAE: {:0.2f} ({:0.1f}%)'.format(mae_rsd, mae_rsd_p)
    str_rsu_toa = 'MAE, TOA: {:0.2f} ({:0.1f}%)'.format(mae_rsu_toa, 
                                100 * np.abs(mae_rsu_toa / rsu_toa_true.mean()))
    str_rsu_sfc = 'MAE, sfc: {:0.2f} ({:0.1f}%)'.format(mae_rsu_sfc, 
                                100 * np.abs(mae_rsu_sfc / rsu_sfc_true.mean()))
    str_rsd_sfc = 'MAE, sfc: {:0.2f} ({:0.1f}%)'.format(mae_rsd_sfc, 
                                100 * np.abs(mae_rsd_sfc / rsd_sfc_true.mean()))
    errfunc = mae
    hr_err      = errfunc(dTdt_true, dTdt_pred)
    fluxup_err  = errfunc(fluxup_true, fluxup_pred)
    fluxdn_err  = errfunc(fluxdn_true, fluxdn_pred)
    ylay    = 0.01*pres_lay.mean(axis=0)
    y       = 0.01*pres.mean(axis=0)

    try:
        ax[0]
    except:
        fig, ax = plt.subplots(ncols=3)
    ax[0].plot(fluxdn_err,  y)
    ax[1].plot(fluxup_err,  y)
    ax[0].set_ylabel('Pressure (hPa)',fontsize=ts_ax)
    if xlabel:
        ax[0].set_xlabel('Downwelling flux (W m$^{-2}$)',fontsize=ts_ax); 
        ax[1].set_xlabel('Upwelling flux (W m$^{-2}$)',fontsize=ts_ax); 
        ax[2].set_xlabel('Heating rate (K d$^{-1}$)',fontsize=ts_ax); 
    ax[2].plot(hr_err,  ylay)
    ax[0].invert_yaxis();  ax[1].invert_yaxis(); ax[2].invert_yaxis()
    ax[0].grid(); ax[1].grid(); ax[2].grid()
    if xmax != None:
        ax[0].set_xlim([0,xmax[0]]); ax[1].set_xlim([0,xmax[1]]); ax[2].set_xlim([0,xmax[2]])
    ax[0].annotate(str_rsd, xy=xyc1, xycoords=xycc,size=ts)
    ax[0].annotate(str_rsd_sfc, xy=xyc2, xycoords=xycc,size=ts)
    
    ax[1].annotate(str_rsu, xy=xyc1, xycoords=xycc,size=ts)
    ax[1].annotate(str_rsu_sfc, xy=xyc2, xycoords=xycc,size=ts)
    ax[1].annotate(str_rsu_toa, xy=xyc3, xycoords=xycc,size=ts)

    ax[2].annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
    ax[2].annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
    ax[2].annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
    ax[2].annotate(str4, xy=xyc4, xycoords=xycc,size=ts)
    

def plot_hist2d(y_true,y_pred,nbins,norm):
  
    x = y_true.flatten()
    y = y_pred.flatten()
    err =  np.corrcoef(x,y)[0,1]; err = err**2
    if norm == True:
        fig, ax = plt.subplots()
        (counts, ex, ey, img) = ax.hist2d(x, y, bins=nbins, norm=LogNorm())
    else:
        plt.hist2d(x, y, bins=40)
    if (np.max(x) < 1.1) & (np.min(x) > -0.1):
        plt.xlabel('Transmittance')
        plt.ylabel('Transmittance (predicted)')
    elif (np.min(x) < 0.0):
        plt.xlabel('Normalized optical depth')
        plt.ylabel('Normalized optical depth (predicted)')
    else:
        plt.xlabel('Optical depth')
        plt.ylabel('Optical depth (predicted)')
        
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    ax.set_xlim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    ax.set_ylim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')

    mse_err = mse(x,y)
    mae_err = np.mean(np.abs((x - y)))
    textstr0 = 'R-squared = {:0.5f}'.format(err)
    textstr1 = 'MSE = {:0.5f}'.format(mse_err)
    textstr2 = 'MAE = {:0.5f}'.format(mae_err)
    plt.annotate(textstr0, xy=(-7.0, 0.15), xycoords='axes fraction')
    plt.annotate(textstr1, xy=(-7.0, 0.10), xycoords='axes fraction')
    plt.annotate(textstr2, xy=(-7.0, 0.05), xycoords='axes fraction')
    ax.grid()
    ident = [xmin, xmax]
    ax.plot(ident,ident,'k')
    del x,y


def plot_hist2d_reftrans(y_true,y_pred,nbins,norm):
    # y_true and y_pred have dims (nsamples, noutputs)
    # one plot per output feature (noutputs in total)
    varnames = ['Rdif',      'Tdif','Rdir','Tdir']
    varnames_p = ['Rdif (predicted)', 'Tdif (predicted)','Rdir (predicted)',
               'Tdir (predicted']
    
    varnames_long = ['Diffuse reflectance', 'Diffuse transmittance',
                     'Direct reflectance','Direct transmittance']

    # ny = y_true.shape[1] # 4
    fig, ax = plt.subplots(2, 2)
    i = 0
    for ix in range(2):
        for iy in range(2):
            x = y_true[:,i].flatten()
            y = y_pred[:,i].flatten()
            
            err =  np.corrcoef(x,y)[0,1]; err = err**2

            (counts, ex, ey, img) = ax[ix,iy].hist2d(x, y, bins=nbins, norm=LogNorm())
            
            ax[ix,iy].set_xlabel(varnames[i])
            ax[ix,iy].set_ylabel(varnames_p[i])
            
            ax[ix,iy].set_xlim(0,1)
            ax[ix,iy].set_ylim(0,1)
        
        
            divider = make_axes_locatable(ax[ix,iy])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(img, cax=cax, orientation='vertical')
            
        
            mse_err = mse(x,y)
            mae_err = np.mean(np.abs((x - y)))
            bias_err = bias(x,y)
            textstr0 = 'R-squared = {:0.4f}'.format(err)
            # textstr1 = 'MSE = {:0.5f}'.format(mse_err)
            textstr1 = 'Bias = {:0.5f}'.format(bias_err)
            textstr2 = 'MAE = {:0.5f}'.format(mae_err)
            plt.annotate(textstr0, xy=(-9.0, 0.15), xycoords='axes fraction')
            plt.annotate(textstr1, xy=(-9.0, 0.10), xycoords='axes fraction')
            plt.annotate(textstr2, xy=(-9.0, 0.05), xycoords='axes fraction')
            ax[ix,iy].grid()
            ident = [0, 1]
            ax[ix,iy].plot(ident,ident,'k')
            ax[ix,iy].set_title("{}".format(varnames_long[i]))

            i = i + 1
    del x,y

def plot_hist2d_tau(y_true,y_pred,nbins,norm):
    x = y_true[(y_true<10)&(y_pred<10)].flatten()
    y = y_pred[(y_true<10)&(y_pred<10)].flatten()
    plot_hist2d(x,y,nbins,norm)
    del x,y

def plot_hist2d_T(y_true,y_pred,nbins,norm):
    y_true = np.exp(-y_true).flatten()
    y_pred = np.exp(-y_pred).flatten()
    plot_hist2d(y_true,y_pred,nbins,norm)
    del y_true,y_pred
    
    