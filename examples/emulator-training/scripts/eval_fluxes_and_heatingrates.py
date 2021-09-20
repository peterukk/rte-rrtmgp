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


def bias(predictions, targets,ax):
    return np.mean(predictions-targets,axis=ax)

def calc_heatingrate(fluxup,fluxdn,p):
    
    F = fluxdn - fluxup
    dF = np.gradient(F,axis=1)
    
    dp = np.gradient(p,axis=1)
    dFdp = dF/dp
    
    dTdt = (g/cp)*(dFdp) # K / s
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

dat_ref         =  Dataset(fname_ref)
dat_reftrans    =  Dataset(fname_reftrans)
dat_rrtmgp      =  Dataset(fname_rrtmgp)
dat_rrtmgp_old  =  Dataset(fname_rrtmgp_old)
dat_radscheme   =  Dataset(fname_radscheme)

p   = dat_ref.variables['pres_level'][:,:,:].data
lon = np.rad2deg(dat_ref.variables['clon'][:].data)
lat = np.rad2deg(dat_ref.variables['clat'][:].data)
timedat = dat_ref.variables['time'][:]

ntime = p.shape[0]
nsite = p.shape[1]
nlev = p.shape[2]

shape = (ntime*nsite,nlev)

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

rsu_toa_ref         = dat_ref.variables[var2][:,:,0].data
rsu_toa_reftrans    = dat_reftrans.variables[var2][:,:,0].data
rsu_toa_rrtmgp      = dat_rrtmgp.variables[var2][:,:,0].data
rsu_toa_rrtmgp_old  = dat_rrtmgp_old.variables[var2][:,:,0].data
rsu_toa_radscheme   = dat_radscheme.variables[var2][:,:,0].data

rsd_toa_ref         = dat_ref.variables[var1][:,:,0].data
rsd_sfc_ref         = dat_ref.variables[var1][:,:,-1].data

cloudfrac = dat_ref.variables['cloud_fraction'][:]
sza = dat_ref.variables['solar_zenith_angle'][:]

nightcols = (sza > 89.99)
nightcols_flat = nightcols.flatten()

if exclude_night_cols:
    for var in [fluxdn_ref, fluxup_ref, fluxdn_reftrans, fluxup_reftrans,
            fluxdn_rrtmgp_old, fluxup_rrtmgp_old, fluxdn_rrtmgp, fluxup_rrtmgp,
            fluxdn_radscheme, fluxup_radscheme]:
        var[nightcols_flat] = 0.0
    for var in [rsu_toa_ref, rsu_toa_reftrans, rsu_toa_rrtmgp, rsu_toa_rrtmgp_old, 
                rsu_toa_radscheme, rsd_toa_ref, rsd_sfc_ref]:
        var[nightcols] = 0.0


fluxnet_ref         = fluxdn_ref - fluxup_ref
fluxnet_reftrans    = fluxdn_reftrans - fluxup_reftrans
fluxnet_rrtmgp_old  = fluxdn_rrtmgp_old - fluxup_rrtmgp_old
fluxnet_rrtmgp      = fluxdn_rrtmgp - fluxup_rrtmgp
fluxnet_radscheme    = fluxdn_radscheme - fluxup_radscheme 

dTdt_ref        = calc_heatingrate(fluxup_ref,fluxdn_ref,p)
dTdt_reftrans   = calc_heatingrate(fluxup_reftrans,fluxdn_reftrans,p)
dTdt_rrtmgp_old = calc_heatingrate(fluxup_rrtmgp_old,fluxdn_rrtmgp_old,p)
dTdt_rrtmgp     = calc_heatingrate(fluxup_rrtmgp,fluxdn_rrtmgp,p)
dTdt_radscheme  = calc_heatingrate(fluxup_radscheme, fluxdn_radscheme,p)

dat_ref.close()
dat_reftrans.close()
dat_rrtmgp.close()
dat_rrtmgp_old.close()
dat_radscheme.close()



    
# ESCAPE SLIDE FIGURE

# xx_reftrans     = mae(fluxnet_ref, fluxnet_reftrans,ax=0)
# xx_rrtmgp_old   = mae(fluxnet_ref, fluxnet_rrtmgp_old,ax=0)
# xx_rrtmgp       = mae(fluxnet_ref, fluxnet_rrtmgp ,ax=0)
# xx_radscheme    = mae(fluxnet_ref, fluxnet_radscheme,ax=0)
    
# fs_ax = 14;  
# ind_p = 3 
# pe_radscheme   = 100 * xx_radscheme.mean() / fluxnet_ref.mean()
# pe_rrtmgp      = 100 * xx_rrtmgp.mean() / fluxnet_ref.mean()
# pe_rrtmgp_old  = 100 * xx_rrtmgp_old.mean() / fluxnet_ref.mean()


# label1 = "NN emulating the radiation scheme (error {:0.2f}%)".format(pe_radscheme)
# label2 = "NN emulating gas optics (error {:0.2f}%)".format(pe_rrtmgp)
# label3 = "NN emulating gas optics, 2020 model (error {:0.2f}%)".format(pe_rrtmgp_old)
# figtitle = 'Mean absolute error in net shortwave flux, CAMS 2017'

# fig, ax = plt.subplots(1)
# ax.plot(xx_radscheme[ind_p:], yy[ind_p:],'r', label=label1)
# ax.plot(xx_rrtmgp[ind_p:], yy[ind_p:],'b',    label=label2)
# ax.plot(xx_rrtmgp_old[ind_p:], yy[ind_p:],'b',label=label3, ls='--')

# ax.invert_yaxis(); ax.grid()
# ax.set_ylabel('Pressure (hPa)',fontsize=fs_ax)
# ax.set_xlabel('Flux (W m$^{-2}$)',fontsize=fs_ax); 
# fig.suptitle(figtitle, fontsize=fs_bigtitle)
# ax.legend(fontsize=13)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=11)
  
# # xx_reftrans     = bias(fluxnet_ref, fluxnet_reftrans,ax=0)
# # xx_rrtmgp_old    = bias(fluxnet_ref, fluxnet_rrtmgp_old,ax=0)
# # xx_rrtmgp       = bias(fluxnet_ref, fluxnet_rrtmgp ,ax=0)
# # xx_radscheme    = bias(fluxnet_ref, fluxnet_radscheme,ax=0)


# x0_dn_ref          = fluxdn_ref.mean(axis=0)
# x0_up_ref          = fluxup_ref.mean(axis=0)

# # x0_reftrans     = fluxdn_reftrans.mean(axis=0)
# # # x0_rrtmgp_old   = fluxdn_rrtmgp_old,ax=0)
# # x0_rrtmgp       = fluxdn_rrtmgp.mean(axis=0)
# # x0_radscheme    = fluxdn_radscheme.mean(axis=0)
    
# mae_dn_reftrans     = mae(fluxdn_ref, fluxdn_reftrans,ax=0)
# mae_dn_rrtmgp       = mae(fluxdn_ref, fluxdn_rrtmgp,ax=0)
# mae_dn_radscheme    = mae(fluxdn_ref, fluxdn_radscheme,ax=0)

# mae_up_reftrans     = mae(fluxup_ref, fluxup_reftrans,ax=0)
# mae_up_rrtmgp       = mae(fluxup_ref, fluxup_rrtmgp,ax=0)
# mae_up_radscheme    = mae(fluxup_ref, fluxup_radscheme,ax=0)





# DOWENWELLING AND UPWELLING FLUX

yy = 0.01*p[:,:].mean(axis=0)

l_mae = 'Mean absolute error'
l_bias = 'Mean error (bias)'


fluxes = [[fluxdn_radscheme, fluxup_radscheme], 
          [fluxdn_reftrans,  fluxup_reftrans],
          [fluxdn_rrtmgp,   fluxup_rrtmgp]]
    
fluxes_ref = [fluxdn_ref, fluxup_ref]


xlim = [-15, 15]
fs_cols = 14
fs_ax   = 13
xyc1 = (0.05, 0.87)
xyc2 = (0.05, 0.77)
xyc3 = (0.05, 0.67)
ts = 12
xycc = 'axes fraction'

ind_p = 2

ncols = 2
nrows = 3
fig, ax = plt.subplots(nrows,ncols)

for i in range(nrows): # different models
    for j in range(ncols): # downwelling, then shortwelling

    # j = 0 # downwelling

        # diff = fluxes_dn[i] - fluxdn_ref
        flux    = fluxes[i][j]
        fluxref = fluxes_ref[j]
        diff = flux - fluxref
        
        p5  = np.percentile(diff, 5, axis=0)
        p95 = np.percentile(diff, 95, axis=0)
        xb  = np.mean(diff, axis=0)
        x = np.mean(np.abs(diff),axis=0)
        
        ax[i,j].plot(x[ind_p:], yy[ind_p:],'b', label=l_mae, ls='--')
        ax[i,j].plot(xb[ind_p:], yy[ind_p:],'b', label=l_bias)    
        ax[i,j].fill_betweenx(yy[ind_p:], p5[ind_p:], p95[ind_p:], color='b', alpha=.1)
        ax[i,j].invert_yaxis(); ax[i,j].grid()
        ax[i,j].set_xlim(xlim)
        
        if (i==1 and j==0):  ax[i,j].legend(fontsize=10)
        
        if (i==0): 
            if (j==0):
                ax[i,j].set_title('Total downwelling shortwave flux', fontsize=fs_cols)
            else:
                ax[i,j].set_title('Upwelling shortwave flux', fontsize=fs_cols)

        str2= 'Bias: {:0.2f} ({:0.1f}%)'.format(xb.mean(), 100*xb.mean()/fluxref.mean())
        str1= 'MAE : {:0.2f} ({:0.1f}%)'.format(x.mean(), 100*x.mean()/fluxref.mean())
        str3= 'RMSE: {:0.2f}'.format(rmse_tot(flux, fluxref))
    
        ax[i,j].annotate(str1, xy=xyc1, xycoords=xycc,size=ts)
        ax[i,j].annotate(str2, xy=xyc2, xycoords=xycc,size=ts)
        ax[i,j].annotate(str3, xy=xyc3, xycoords=xycc,size=ts)
        
        if (j==0): ax[i,j].set_ylabel('Pressure (hPa)',fontsize=fs_ax)
        if (i==2): ax[i,j].set_xlabel('Flux (W m$^{-2}$)',fontsize=fs_ax); 
        

xpad = 70
ypad = 40
rowtitles = ['NN-Radscheme', 'NN-RefTrans', 'NN-RRTMGP']

for axx, row in zip(ax[:,0], rowtitles):
    axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad - xpad, 0 + ypad),
                xycoords=axx.yaxis.label, textcoords='offset points',
                size=14, ha='right', va='center', rotation=45)
                



#  HEATING RATES


fs_bigtitle = 16
fs_ax = 14;  
ind_p = 20

# errfunc = mae
errfunc = rmse
hre_radscheme       = errfunc(dTdt_ref[:,ind_p:], dTdt_radscheme[:,ind_p:])
hre_reftrans        = errfunc(dTdt_ref[:,ind_p:], dTdt_reftrans[:,ind_p:])
hre_rrtmgp          = errfunc(dTdt_ref[:,ind_p:], dTdt_rrtmgp[:,ind_p:])
hre_rrtmgp_old      = errfunc(dTdt_ref[:,ind_p:], dTdt_rrtmgp_old[:,ind_p:])


refvar = np.abs(dTdt_ref[:,ind_p:].mean())
pe_radscheme   = 100 * hre_radscheme.mean() / refvar
pe_rrtmgp      = 100 * hre_rrtmgp.mean() / refvar
pe_reftrans     = 100 * hre_reftrans.mean() / refvar
pe_rrtmgp_old  = 100 * hre_rrtmgp_old.mean() / refvar


label1 = "NN emulating the radiation scheme (error {:0.2f}%)".format(pe_radscheme)
label2 = "NN emulating reflectance-transmittance computations (error {:0.2f}%)".format(pe_reftrans)
label3 = "NN emulating gas optics (error {:0.2f}%)".format(pe_rrtmgp)
label4 = "NN emulating gas optics, 2020 model (error {:0.2f}%)".format(pe_rrtmgp_old)
figtitle = 'Mean absolute error in shortwave heating rate'

fig, ax = plt.subplots(1)
ax.plot(hre_radscheme,  yy[ind_p:],'r', label=label1)
ax.plot(hre_reftrans,   yy[ind_p:],'black', label=label2)
ax.plot(hre_rrtmgp,     yy[ind_p:],'b', label=label3)
ax.plot(hre_rrtmgp_old,     yy[ind_p:],'b', label=label4, ls='--')

ax.invert_yaxis(); ax.grid()
ax.set_ylabel('Pressure (hPa)',fontsize=fs_ax)
ax.set_xlabel('Heating rate (W m$^{-2}$)',fontsize=fs_ax); 
fig.suptitle(figtitle, fontsize=fs_bigtitle)
ax.legend(fontsize=13)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=11)

xmax = 0.8
ax.set_xlim(0, xmax)


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

# import psyplot.project as psy    
# from psyplot import open_dataset
# from psy_maps.plotters import FieldPlotter

# fname = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/fluxes/REF.nc"

# ds = open_dataset(fname_ref)

# maps = psy.plot.mapplot(ds, name='surface_temperature', projection='moll',
#                         cmap='RdBu_r',  title='2-metre temperature',clabel="Temperature (K)")

# clon = np.rad2deg(ds['clon'].data)
# clat = np.rad2deg(ds['clat'].data)

# z = ds['rsu'][0,0,:].data




fig = plt.figure(figsize=(16,9))
fig.suptitle('Upwelling flux at top-of-atmosphere', fontsize = fs_bigtitle)

ax0 = plt.subplot(4, 1, 1,projection=proj)
cs0 = ax0.tricontourf(x, y, z0)
ax0.coastlines(color='white')

cax0= ax0.inset_axes([1.15 ,0.13, 0.08, 0.8], transform=ax0.transAxes)
colorbar0 = fig.colorbar(cs0, cax=cax0)
# colorbar0.ax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11)
cax0.set_xlabel('Flux (W m$^{-2}$)',fontsize=11,  labelpad=11)

ax1 = plt.subplot(4, 1, 2,projection=proj)
cs1 = ax1.tricontourf(x, y, z1, levels=bounds)
ax1.coastlines(color='white')


ax2 = plt.subplot(4, 1, 3,projection=proj)
cs2 = ax2.tricontourf(x, y, z2, levels=bounds)
ax2.coastlines(color='white')

cax = ax2.inset_axes([1.15, -0.6, 0.08, 2.5], transform=ax2.transAxes)
colorbar = fig.colorbar(cs2, cax=cax)
cax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11, labelpad=11)


ax3 = plt.subplot(4, 1, 4, projection=proj)
cs3 = ax3.tricontourf(x, y, z3, levels=bounds)
ax3.coastlines(color='white')

xpad = 110
ypad = 40
rowtitles = ['(a) TOA flux', '(b) NN-RadScheme', '(c) NN-RefTrans', '(d) NN-RRTMGP']

for axx, row in zip([ax0,ax1,ax2,ax3], rowtitles):
    axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad +xpad, 0 + ypad),
                xycoords=axx.yaxis.label, textcoords='offset points',
                size=14, ha='right', va='center', rotation=45)
                






z0 = rsu_toa_ref.mean(axis=0)
# z0 = rsd_sfc_ref.mean(axis=0)


z0 = sza[7,:]
# z0 = sza.mean(axis=0)
# z0 = np.mean(cloudfrac.sum(axis=2),axis=0)

x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T

fig = plt.figure(figsize=(16,9))
proj = ccrs.EqualEarth()

proj = ccrs.EqualEarth()
ax = plt.axes(projection=proj)
cs = ax.tricontourf(x, y, z0)
ax.coastlines(color='white')

cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
colorbar = fig.colorbar(cs, cax=cax)


