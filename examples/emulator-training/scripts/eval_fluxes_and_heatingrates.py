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

# def toa_lbl(fluxup_ref,fluxup_nn,fluxup_lbl, ind=5):

#     fluxup_ref1 = fluxup_ref[ind,:,:]
#     fluxup_nn1 = fluxup_nn[ind,:,:]
#     fluxup_lbl1 = fluxup_lbl[ind,:,:]

#     fluxup_ref1_TOA = np.sum(prof_weight[:,0]*fluxup_ref1[:,0])
#     fluxup_nn1_TOA = np.sum(prof_weight[:,0]*fluxup_nn1[:,0])
#     fluxup_lbl1_TOA = np.sum(prof_weight[:,0]*fluxup_lbl1[:,0])

#     fluxup_ref1_TOA_test = 100/15 * np.sum(prof_weight[inds_test,0]*fluxup_ref1[inds_test,0])
#     fluxup_nn1_TOA_test = 100/15 * np.sum(prof_weight[inds_test,0]*fluxup_nn1[inds_test,0])
#     fluxup_lbl1_TOA_test = 100/15 * np.sum(prof_weight[inds_test,0]*fluxup_lbl1[inds_test,0])

#     print("comparing top-of-the-atmosphere upwelling LW flux for {}".format(expt_label[ind]))
#     print( "TOA forcing for RRTMGP-REF: {:0.3f} W/m2 (GLOBAL), {:0.3f} W/m2 (TEST-sites)".format(fluxup_ref1_TOA,fluxup_ref1_TOA_test))
#     print( "TOA forcing for LineByLine: {:0.3f} W/m2 (GLOBAL), {:0.3f} W/m2 (TEST-sites)".format(fluxup_lbl1_TOA, fluxup_lbl1_TOA_test))
#     print( "TOA forcing for RRTMGP-NN : {:0.3f} W/m2 (GLOBAL), {:0.3f} W/m2 (TEST-sites)".format(fluxup_nn1_TOA, fluxup_nn1_TOA_test))
#     return fluxup_ref1_TOA,fluxup_nn1_TOA,fluxup_lbl1_TOA



def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def rmse_prof(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=0))

def rmse_tot(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions,targets,ax):
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



rootdir = "../fluxes/"

var1 = 'rsd'
var2 = 'rsu'

fname_ref       = rootdir+'rsud_CAMS_2018_REF.nc'
fname_reftrans  = rootdir+'rsud_CAMS_2018_REFTRANS.nc'
fname_rrtmgp    = rootdir+'rsud_CAMS_2018_RRTMGP.nc'
fname_rrtmgp_old = rootdir+'rsud_CAMS_2018_RRTMGP_2020.nc'
fname_radscheme = rootdir+'rsud_CAMS_2018_RADSCHEME.nc'

dat_ref         =  Dataset(fname_ref)
dat_reftrans    =  Dataset(fname_reftrans)
dat_rrtmgp      =  Dataset(fname_rrtmgp)
dat_rrtmgp_old  =  Dataset(fname_rrtmgp_old)
dat_radscheme   =  Dataset(fname_radscheme)

p   = dat_ref.variables['plev'][0,:,:].data
lon = dat_ref.variables['lon'][:].data
lat = dat_ref.variables['lat'][:].data
timedat = dat_ref.variables['time'][0:8].data

nsite = p.shape[0]
nlay = p.shape[1]

ntime = 8
nlat = 25
nlon = 24

latt = np.reshape(lat,(nlon,nlat,ntime))[:,:,0]
lonn = np.reshape(lon,(nlon,nlat,ntime))[:,:,0]


fluxdn_ref          = dat_ref.variables[var1][0,:,:].data
fluxup_ref          = dat_ref.variables[var2][0,:,:].data
fluxdn_reftrans     = dat_reftrans.variables[var1][0,:,:].data
fluxup_reftrans     = dat_reftrans.variables[var2][0,:,:].data
fluxdn_rrtmgp       = dat_rrtmgp.variables[var1][0,:,:].data
fluxup_rrtmgp       = dat_rrtmgp.variables[var2][0,:,:].data
fluxdn_rrtmgp_old   = dat_rrtmgp_old.variables[var1][0,:,:].data
fluxup_rrtmgp_old   = dat_rrtmgp_old.variables[var2][0,:,:].data
fluxdn_radscheme    = dat_radscheme.variables[var1][0,:,:].data
fluxup_radscheme    = dat_radscheme.variables[var2][0,:,:].data



fluxnet_ref         = fluxdn_ref - fluxup_ref
fluxnet_reftrans    = fluxdn_reftrans - fluxup_reftrans
fluxnet_rrtmgp      = fluxdn_rrtmgp - fluxup_rrtmgp
fluxnet_rrtmgp_old      = fluxdn_rrtmgp_old - fluxup_rrtmgp_old
fluxnet_radscheme    = fluxdn_radscheme - fluxup_radscheme 

dTdt_ref        = calc_heatingrate(fluxup_ref,fluxdn_ref,p)
dTdt_reftrans   = calc_heatingrate(fluxup_reftrans,fluxdn_reftrans,p)
dTdt_rrtmgp     = calc_heatingrate(fluxup_rrtmgp,fluxdn_rrtmgp,p)
dTdt_rrtmgp_old = calc_heatingrate(fluxup_rrtmgp_old,fluxdn_rrtmgp_old,p)
dTdt_radscheme  = calc_heatingrate(fluxup_radscheme, fluxdn_radscheme,p)




# # ---------- TEST SITES, REGULAR Y AXIS ---------

# ind_p = 8

# fs_ax = 13; fs_leg = 13; fs_subtitle = 15; fs_bigtitle = 16

# label1 = 'NN-RRTMGP'
# label2 = 'NN-REFTRANS'
# label3 = 'NN-RADSCHEME'

# yy = 0.01*p[:,:].mean(axis=0)

# var = 'rsd'
# figtitle = 'Total downwelling shortwave'

# var = 'rsu'
# figtitle = 'Upwelling shortwave'

# xx_reftrans = mae(dat_ref.variables[var][0,:,:].data, dat_reftrans.variables[var][0,:,:].data,ax=0)
# xx_rrtmgp   = mae(dat_ref.variables[var][0,:,:].data, dat_rrtmgp.variables[var][0,:,:].data,ax=0)
# xx_radscheme = mae(dat_ref.variables[var][0,:,:].data, dat_radscheme.variables[var][0,:,:].data,ax=0)

# xmax = np.max((xx_reftrans.max(),xx_rrtmgp.max()))

# fig, (ax,ax2,ax3) = plt.subplots(1,3)

# ax.plot(xx_rrtmgp[ind_p:], yy[ind_p:],'b',label=label1)
# # ax2.plot(xx_reftrans[ind_p:], yy[ind_p:],'b',label=label2)
# ax2.plot(xx_rrtmgp_old[ind_p:], yy[ind_p:],'b',label=label2)
# ax3.plot(xx_radscheme[ind_p:], yy[ind_p:],'b',label=label3)

# ax.set_ylabel('Pressure (hPa)',fontsize=fs_ax)
# fig.suptitle(figtitle, fontsize=fs_bigtitle)

# for axx in (ax,ax2,ax3):
#     axx.invert_yaxis(); axx.grid()
#     # axx.set_xlim([0,xmax]);
#     # ax.set_xlabel('K d$^{-1}$',fontsize=fs_ax); 
#     axx.set_xlabel('Flux (W m$^{-2}$)',fontsize=fs_ax); 
    
    
xx_reftrans     = mae(fluxnet_ref, fluxnet_reftrans,ax=0)
xx_rrtmgp_old   = mae(fluxnet_ref, fluxnet_rrtmgp_old,ax=0)
xx_rrtmgp       = mae(fluxnet_ref, fluxnet_rrtmgp ,ax=0)
xx_radscheme    = mae(fluxnet_ref, fluxnet_radscheme,ax=0)
    
fs_ax = 14;  
ind_p = 3 
pe_radscheme   = 100 * xx_radscheme.mean() / fluxnet_ref.mean()
pe_rrtmgp      = 100 * xx_rrtmgp.mean() / fluxnet_ref.mean()
pe_rrtmgp_old  = 100 * xx_rrtmgp_old.mean() / fluxnet_ref.mean()


label1 = "NN emulating the radiation scheme (error {:0.2f}%)".format(pe_radscheme)
label2 = "NN emulating gas optics (error {:0.2f}%)".format(pe_rrtmgp)
label3 = "NN emulating gas optics, 2020 model (error {:0.2f}%)".format(pe_rrtmgp_old)
figtitle = 'Mean absolute error in net shortwave flux, CAMS 2017'

fig, ax = plt.subplots(1)
ax.plot(xx_radscheme[ind_p:], yy[ind_p:],'r', label=label1)
ax.plot(xx_rrtmgp[ind_p:], yy[ind_p:],'b',    label=label2)
ax.plot(xx_rrtmgp_old[ind_p:], yy[ind_p:],'b',label=label3, ls='--')

ax.invert_yaxis(); ax.grid()
ax.set_ylabel('Pressure (hPa)',fontsize=fs_ax)
ax.set_xlabel('Flux (W m$^{-2}$)',fontsize=fs_ax); 
fig.suptitle(figtitle, fontsize=fs_bigtitle)
ax.legend(fontsize=13)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=11)
  

# xx_reftrans     = bias(fluxnet_ref, fluxnet_reftrans,ax=0)
# xx_rrtmgp_old    = bias(fluxnet_ref, fluxnet_rrtmgp_old,ax=0)
# xx_rrtmgp       = bias(fluxnet_ref, fluxnet_rrtmgp ,ax=0)
# xx_radscheme    = bias(fluxnet_ref, fluxnet_radscheme,ax=0)


x0_dn_ref          = fluxdn_ref.mean(axis=0)
x0_up_ref          = fluxup_ref.mean(axis=0)

# x0_reftrans     = fluxdn_reftrans.mean(axis=0)
# # x0_rrtmgp_old   = fluxdn_rrtmgp_old,ax=0)
# x0_rrtmgp       = fluxdn_rrtmgp.mean(axis=0)
# x0_radscheme    = fluxdn_radscheme.mean(axis=0)
    
mae_dn_reftrans     = mae(fluxdn_ref, fluxdn_reftrans,ax=0)
mae_dn_rrtmgp       = mae(fluxdn_ref, fluxdn_rrtmgp,ax=0)
mae_dn_radscheme    = mae(fluxdn_ref, fluxdn_radscheme,ax=0)

mae_up_reftrans     = mae(fluxup_ref, fluxup_reftrans,ax=0)
mae_up_rrtmgp       = mae(fluxup_ref, fluxup_rrtmgp,ax=0)
mae_up_radscheme    = mae(fluxup_ref, fluxup_radscheme,ax=0)



# DOWENWELLING AND UPWELLING FLUX

yy = 0.01*p[:,:].mean(axis=0)

l_mae = 'Mean absolute error'
l_bias = 'Mean error (bias)'


fluxes = [[fluxdn_radscheme, fluxup_radscheme], 
          [fluxdn_rrtmgp,   fluxup_rrtmgp],
          [fluxdn_reftrans,  fluxup_reftrans]]
    
fluxes_ref = [fluxdn_ref, fluxup_ref]


xlim = [-30, 30]
fs_cols = 14
fs_ax   = 13
xyc1 = (0.05, 0.87)
xyc2 = (0.05, 0.77)
xyc3 = (0.05, 0.67)
ts = 12
xycc = 'axes fraction'

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
        
        p5  = np.percentile(diff, 5, axis=0); p95 = np.percentile(diff, 95, axis=0)
        xb  = np.mean(diff, axis=0); x = np.mean(np.abs(diff),axis=0)
        
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
rowtitles = ['NN-Radscheme', 'NN-RRTMGP','NN-RefTrans']

for axx, row in zip(ax[:,0], rowtitles):
    axx.annotate(row, xy=(5, 0.5), xytext=(-axx.yaxis.labelpad - xpad, 0 + ypad),
                xycoords=axx.yaxis.label, textcoords='offset points',
                size=14, ha='right', va='center', rotation=45)
                

   
# from mpl_toolkits.basemap import Basemap
# from matplotlib.colors import DivergingNorm
   
# # ----------------

# # Plot a surface or TOA variable

# var = np.abs(fluxdn_ref-fluxdn_rrtmgp)[:,-1]
# var = np.reshape(var,(nlon,nlat,ntime)).mean(axis=2)

# cmin = 0; cmax = 1
# fs = 14; fs_text = 10
# jet = plt.cm.get_cmap('RdBu_r')
# # jet = plt.cm.get_cmap('rainbow')

# fig = plt.figure(constrained_layout=False)
# gs  = fig.add_gridspec(nrows=4, ncols=2)

# i = 0

# # Plot 1
# f_ax1 = fig.add_subplot(gs[i, :])
# m = Basemap(projection='moll',lon_0=0,resolution='c',ax=f_ax1)

# m.pcolormesh(lonn, latt, var, latlon=True, cmap='RdBu_r')
# x,y = m(lonn, latt)
# im1 = m.pcolormesh(x, y, var, latlon=False, cmap=jet, vmin = cmin, vmax = cmax)

# cbar =  m.colorbar(im1)
# cbar.set_clim(-2.0, 2.0)
# # f_ax1.set_title('RRTMGP-NN - RRTMGP' , fontsize=fs)

# # cbar.set_label('W m$^{-2}$', fontsize = fs-3)
# # labelstr = '(b)'

# m.drawparallels(np.arange(-90.,91.,30.))
# m.drawmeridians(np.arange(-180.,181.,60.))
# m.drawcoastlines()



dat_ref.close()
dat_reftrans.close()
dat_rrtmgp.close()
dat_radscheme.close()