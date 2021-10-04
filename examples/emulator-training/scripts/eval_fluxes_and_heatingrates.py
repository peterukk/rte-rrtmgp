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
import num2tex

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
# fname_radscheme = rootdir+'tmp.nc'
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
fluxdn_radscheme_r    = dat_radscheme_r.variables[var1][:,:,:].data.reshape(shape)
fluxup_radscheme_r    = dat_radscheme_r.variables[var2][:,:,:].data.reshape(shape)

rsu_toa_ref         = dat_ref.variables[var2][:,:,0].data
rsu_toa_reftrans    = dat_reftrans.variables[var2][:,:,0].data
rsu_toa_rrtmgp      = dat_rrtmgp.variables[var2][:,:,0].data
rsu_toa_rrtmgp_old  = dat_rrtmgp_old.variables[var2][:,:,0].data
rsu_toa_radscheme   = dat_radscheme.variables[var2][:,:,0].data
rsu_toa_radscheme_r = dat_radscheme_r.variables[var2][:,:,0].data

rsd_toa_ref         = dat_ref.variables[var1][:,:,0].data
rsd_sfc_ref         = dat_ref.variables[var1][:,:,-1].data

cloudfrac = dat_ref.variables['cloud_fraction'][:]
sza = dat_ref.variables['solar_zenith_angle'][:]

nightcols = (sza > 89.99)
nightcols_flat = nightcols.flatten()

if exclude_night_cols:
    for var in [fluxdn_ref, fluxup_ref, fluxdn_reftrans, fluxup_reftrans,
            fluxdn_rrtmgp_old, fluxup_rrtmgp_old, fluxdn_rrtmgp, fluxup_rrtmgp,
            fluxdn_radscheme, fluxup_radscheme, fluxdn_radscheme_r, fluxup_radscheme_r]:
        var[nightcols_flat] = 0.0
    for var in [rsu_toa_ref, rsu_toa_reftrans, rsu_toa_rrtmgp, rsu_toa_rrtmgp_old, 
                rsu_toa_radscheme, rsu_toa_radscheme_r, rsd_toa_ref, rsd_sfc_ref]:
        var[nightcols] = 0.0


fluxnet_ref         = fluxdn_ref - fluxup_ref
fluxnet_reftrans    = fluxdn_reftrans - fluxup_reftrans
fluxnet_rrtmgp_old  = fluxdn_rrtmgp_old - fluxup_rrtmgp_old
fluxnet_rrtmgp      = fluxdn_rrtmgp - fluxup_rrtmgp
fluxnet_radscheme    = fluxdn_radscheme - fluxup_radscheme 
fluxnet_radscheme_r  = fluxdn_radscheme - fluxup_radscheme 

dTdt_ref        = calc_heatingrate(fluxup_ref,fluxdn_ref,p)
dTdt_reftrans   = calc_heatingrate(fluxup_reftrans,fluxdn_reftrans,p)
dTdt_rrtmgp_old = calc_heatingrate(fluxup_rrtmgp_old,fluxdn_rrtmgp_old,p)
dTdt_rrtmgp     = calc_heatingrate(fluxup_rrtmgp,fluxdn_rrtmgp,p)
dTdt_radscheme  = calc_heatingrate(fluxup_radscheme, fluxdn_radscheme,p)
dTdt_radscheme_r = calc_heatingrate(fluxup_radscheme_r, fluxdn_radscheme_r,p)

dat_ref.close()
dat_reftrans.close()
dat_rrtmgp.close()
dat_rrtmgp_old.close()
dat_radscheme.close()
dat_radscheme_r.close()




# PROFILES OF FLUX AND HEATING RATE ERRORS

yy = 0.01*p[:,:].mean(axis=0)

l_mae = 'Mean absolute error'
l_bias = 'Mean error (bias)'


fluxes = [[fluxdn_radscheme, fluxup_radscheme], 
          [fluxdn_radscheme_r, fluxup_radscheme_r],
          [fluxdn_reftrans,  fluxup_reftrans],
          [fluxdn_rrtmgp,   fluxup_rrtmgp]]
    
fluxes_ref = [fluxdn_ref, fluxup_ref]

heatingrates = [dTdt_radscheme, dTdt_radscheme_r, dTdt_reftrans, dTdt_rrtmgp]

xlim = [-10, 10]
xlim2 = [-1.5, 1.5]
fs_cols = 14
fs_ax   = 13
ts = 10
xb = 0.57
xyc1 = (xb, 0.87)
xyc2 = (xb, 0.79)
xyc3 = (xb, 0.71)
xyc4 = (xb, 0.63)
xycc = 'axes fraction'
ind_p = 2

ncols = 3 # metrics
nrows = 4 # models
fig, ax = plt.subplots(nrows,ncols)#, sharey=True)

for i in range(nrows): # different models
    for j in range(2): # downwelling, then shortwelling
        var    = fluxes[i][j]; varref = fluxes_ref[j]
        diff = var - varref
        
        p5  = np.percentile(diff, 5, axis=0); p95 = np.percentile(diff, 95, axis=0)
        xb  = np.mean(diff, axis=0)
        x = np.mean(np.abs(diff),axis=0)
        
        ax[i,j].plot(x[ind_p:], yy[ind_p:],'b', label=l_mae, ls='--')
        ax[i,j].plot(xb[ind_p:], yy[ind_p:],'b', label=l_bias)    
        ax[i,j].fill_betweenx(yy[ind_p:], p5[ind_p:], p95[ind_p:], color='b', alpha=.1)
        ax[i,j].set_xlim(xlim)
        
        if (i==1 and j==0):  ax[i,j].legend(fontsize=10,loc='upper left')
        
        if (i==0): 
            if (j==0):
                ax[i,j].set_title('Total downwelling shortwave flux', fontsize=fs_cols)
            else:
                ax[i,j].set_title('Upwelling shortwave flux', fontsize=fs_cols)

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
    diff = var - varref
    
    p5  = np.percentile(diff, 5, axis=0)
    p95 = np.percentile(diff, 95, axis=0)
    xb  = np.mean(diff, axis=0)
    x = np.mean(np.abs(diff),axis=0)
    
    ax[i,j].plot(x[ind_p:], yy[ind_p:],'b', label=l_mae, ls='--')
    ax[i,j].plot(xb[ind_p:], yy[ind_p:],'b', label=l_bias)    
    ax[i,j].fill_betweenx(yy[ind_p:], p5[ind_p:], p95[ind_p:], color='b', alpha=.1)
    ax[i,j].set_xlim(xlim2)
    
    if (i==1 and j==0):  ax[i,j].legend(fontsize=10)
    
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
ypad = 40
rowtitles = ['FNN-Radscheme', 'RNN-RadScheme', 'FNN-RefTrans', 'FNN-RRTMGP']

for axx, row in zip(ax[:,0], rowtitles):
    axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad - xpad, 0 + ypad),
                xycoords=axx.yaxis.label, textcoords='offset points',
                size=14, ha='right', va='center', rotation=45)
                

def annotate_axis(ax,var,var_ref, xyc1, xyc2, xycc, ts):
        diff = np.abs(var-var_ref)
        varbias = (var-var_ref).mean() 
        str2= 'MAE : {:0.2f} ({:0.1f}%)'.format(diff.mean(), np.abs(100*diff.mean()/var_ref.mean()))
        str1= 'Bias: {:0.2f} ({:0.1f}%)'.format(varbias, 100*varbias/var_ref.mean())

        ax.annotate(str1, xy=xyc1, xycoords=xycc,size=ts,color='k')
        ax.annotate(str2, xy=xyc2, xycoords=xycc,size=ts,color='k')



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

fs_bigtitle = 16
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
xb = 1.02
xyc1 = (xb, 0.87)
xyc2 = (xb, 0.77)
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

# cax = ax3.inset_axes(xvals_cbar, transform=ax3.transAxes)
# colorbar = fig.colorbar(cs3, cax=cax) 
# cax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11, labelpad=11)

ax4 = plt.subplot(nrows, 1, i, projection=proj)
cs4 = ax4.tricontourf(x, y, z4, levels=bounds, cmap=cmap)
ax4.coastlines(color=coastcolor)
annotate_axis(ax4,rsu_toa_rrtmgp,rsu_toa_ref, xyc1, xyc2, xycc, ts2)
i = i +1

cax = ax4.inset_axes(xvals_cbar2, transform=ax4.transAxes)
colorbar = fig.colorbar(cs4, cax=cax,orientation='horizontal') 
cax.set_xlabel('Flux (W m$^{-2}$)',fontsize=11, labelpad=11)

xpad = 110
xpad = 125
# xpad = 145

ypad = 40
ypad = 35

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
      size=14, ha='right', va='center', rotation=0)      






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


