#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 17:00:33 2021

@author: peter
"""

import os
from netCDF4 import Dataset,num2date
import numpy as np
this_dir    = os.getcwd() + "/"
os.chdir(this_dir)
root_dir    = this_dir + "../../" # emulator-training directory
# root_dir = '/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/'
from hum import esat, mixr2rh


# Input data file
fpath = "/media/peter/samsung/data/CAMS/CAMS_2015_RFMIPstyle.nc"

dat         = Dataset(fpath,'a')

p   = dat.variables['pres_level'][:,:,:].data
lon = np.rad2deg(dat.variables['clon'][:].data)
lat = np.rad2deg(dat.variables['clat'][:].data)
timedat = dat.variables['time'][:]

ntime = p.shape[0]
nsite = p.shape[1]
nlev = p.shape[2]


# save solar zenith angle
from sunposition import sunpos
t_unit =  dat.variables['time'].units
t_cal  =  dat.variables['time'].calendar

lonn = lon.reshape(1,nsite).repeat(ntime,axis=0)
latt = lat.reshape(1,nsite).repeat(ntime,axis=0)
lonn = lonn.reshape(nsite*ntime); latt = latt.reshape(nsite*ntime)

timedatt = dat.variables['time'][:].reshape(ntime,1).repeat(nsite,axis=1)
timedatt = timedatt.reshape(nsite*ntime)
times = num2date(timedatt,units = t_unit,calendar = t_cal).data
az,zen = sunpos(times.data,latt,lonn,0)[:2] #discard RA, dec, H

sza_new = zen.reshape(ntime,nsite)
sza = dat.variables['solar_zenith_angle']
sza[:] = sza_new

dat.close()


z1 = sza_new[0,:]

fig = plt.figure(figsize=(16,9))

proj = ccrs.EqualEarth()

x, y, _ = proj.transform_points(ccrs.PlateCarree(), lon, lat).T

# ax = plt.axes(projection=proj)
ax1 = plt.axes(projection=proj)
cs1 = ax1.tricontourf(x, y, z1)
ax1.coastlines(color='white')