#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:43:11 2019

@author: pepe
"""

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


rootdir = '/media/peter/samlinux/gdrive/phd/soft/python/nn-rrtmgp-new/'


fpath_rfmip = rootdir+'inputs_RFMIP.nc'
fpath_ckdmip = rootdir +'ckdmip_evaluation1_concentrations.nc'

fpath_cams = '/media/peter/samlinux/data/CAMS/CAMS_2011.nc'
fpath_cams2 = '/media/peter/samlinux/data/CAMS/N2O/CAMS_n2o_2011.nc'
fpath_new = rootdir+'inputs_CAMS_2011_RFMIPstyle.nc'

dat = Dataset(fpath_cams)
dat_rfmip = Dataset(fpath_rfmip)
dat_ckdmip = Dataset(fpath_ckdmip)

lon         = dat.variables['lon'][:].data; nlon = lon.size
lat         = dat.variables['lat'][:].data; nlat = lat.size
nlay        = dat.variables['lev'][:].data.size
ntime       = dat.variables['time'][:].data.size
# nexpt_old = dat_rfmip.variables['methane_GM'][:].data.size

# p           = dat.variables['pressure'][:].data; 
# sp          = dat.variables['sp'][:].data;
# temp        = dat.variables['t'][:].data;
# t2m         = dat.variables['t2m'][:].data

# NEW FILE
dat_new = Dataset(fpath_new,'w')

nsite = lon.size * lat.size * ntime
nexpt_new = 1

dat_new.createDimension('layer',nlay)
dat_new.createDimension('level',nlay+1)
dat_new.createDimension('site',nsite)
dat_new.createDimension('expt',nexpt_new)

# Longitude latitude information into 1D (site) array
lonvar  = dat_new.createVariable("lon","f4",("site"))
latvar  = dat_new.createVariable("lat","f4",("site"))

latt,lonn = np.meshgrid(lat,lon)  #  (nlon,nlat) so lat major
lonn = lonn.reshape(nlat*nlon,1).repeat(ntime,axis=1)
latt = latt.reshape(nlat*nlon,1).repeat(ntime,axis=1)
lonn = lonn.reshape(nsite); latt = latt.reshape(nsite)

lonvar[:] = lonn; latvar[:] = latt

# Time 
timevar = dat_new.createVariable("time","f4",("site"))
timevar.units           = dat.variables['time'].units
timevar.calendar        = dat.variables['time'].calendar
timevar.standard_name   = dat.variables['time'].standard_name

# Append time
timedat = dat.variables['time'][:]
timedatt = timedat.reshape(1,ntime).repeat(nlat*nlon,axis=0)
timedatt = timedatt.reshape(nlon*nlat*ntime)
timevar[:] = timedatt

varlist = []
for v in dat.variables:
  # this is the name of the variable.
  # print(v)
  # variable needs to have these two dimensions
  check_lists = ['lat','lon']
  if all(t in dat.variables[v].dimensions for t in check_lists):
    varlist.append(v)

print(varlist)
vars_reshaped = {}

# Reshape and append physical variables
for var in varlist:
  # this is the name of the variable.
  if np.size(dat.variables[var].shape)==4:  # 4-dimensional variable (time, lev, lat, lon)
      
      # Extract variable and reduce it to shape (site,level)
      var_dat = dat.variables[var][:,:,:,:]
      
      # Reorder to (lon,lat,time,lev)
      var_dat = np.swapaxes(var_dat,0,3)  # (lon,lev,lat,time)
      var_dat = np.swapaxes(var_dat,1,2)  # (lon,lat,lev,time)
      var_dat = np.swapaxes(var_dat,2,3)  # (lon,lat,time,lev)
      var_dat = var_dat.reshape(nsite,nlay)
  else: # 3-dimensional variable (time, lat, lon)
      # Extract variable and change it to shape (site)
#      var_dat = dat.variables[var][:,inds_keep_lat,::nth_lon]
      var_dat = dat.variables[var][:,:,:]
      # reshape to (lon,lat,time)
      var_dat = np.swapaxes(var_dat,0,2)  # (lon,lat,time)
      var_dat = var_dat.reshape(nsite)
  vars_reshaped[var] = var_dat

temp_lay    = vars_reshaped['t']
p_lay       = vars_reshaped['pressure']
sp          = vars_reshaped['sp']
temp_sfc    = vars_reshaped['t2m']

# pressure and temperature on LEVELS - needs to be inpterpolated between layers
p_lev       = np.zeros((nsite,nlay+1))
temp_lev    = np.zeros((nsite,nlay+1))

p_lev[:,0] = 0.01
temp_lev[:,0] = 0.977 * temp_lay[:,0]

for i in range(nsite):
    p_lev[i,1:-1] = moving_average(p_lay[i,:],2)
    p_lev[i,nlay] = 0.5*(p_lay[i,nlay-1] + sp[i])
    temp_lev[i,1:-1] = moving_average(temp_lay[i,:],2)
    temp_lev[i,nlay] = 0.5*(temp_lay[i,nlay-1] + temp_sfc[i])    

#pressure
var_play  = dat_new.createVariable("pres_layer","f4",("site","layer")); 
var_p_lev = dat_new.createVariable("pres_level","f4",("site","level")); 
var_play[:] = p_lay; var_p_lev[:] = p_lev

#temperature
# temp_lay =  np.reshape(temp_lay,(1,nsite,nlay))
# var_temp = dat_new.createVariable("temp_layer","f4",("expt","site","layer")); 
var_temp = dat_new.createVariable("temp_layer","f4",("site","layer")); 
var_templev = dat_new.createVariable("temp_level","f4",("site","level")); 
var_temp[:] = temp_lay; var_templev[:] = temp_lev

#surface temperature
var_tempsfc = dat_new.createVariable("surface_temperature","f4",("site"));
var_tempsfc[:] = temp_sfc

var_sfc_emis = dat_new.createVariable("surface_emissivity","f4",("site")); 
var_sfc_emis[:] = 0.5


#EXISTING GASES
# (site,layer                            FROM mass mixing ratio to mole fraction
ch4 = vars_reshaped['ch4_c'].data       * 28.9644 / 16.0425   * 1/float(dat_rfmip.variables['methane_GM'].units)
co  = vars_reshaped['co'].data          * 28.9644 / 28.0101   * 1/float(dat_rfmip.variables['carbon_monoxide_GM'].units)
o3  = vars_reshaped['go3'].data         * 28.9644 / 47.9982   * 1/float(dat_rfmip.variables['ozone'].units)
co2 = vars_reshaped['co2'].data         * 28.9644 / 44.0095    * 1/float(dat_rfmip.variables['carbon_dioxide_GM'].units)
no2 = vars_reshaped['no2'].data         * 28.9644 / 46.0055   * 1/1e-6
q   = vars_reshaped['q'].data           * 28.9644 / 18.01528  * 1/float(dat_rfmip.variables['water_vapor'].units)

ch4 = np.reshape(ch4,(nsite,nlay))
co  = np.reshape(co,(nsite,nlay))
o3  = np.reshape(o3,(nsite,nlay))
co2 = np.reshape(co2,(nsite,nlay))
no2 = np.reshape(no2,(nsite,nlay))
q   = np.reshape(q,(nsite,nlay)); q[q<0] = 0.0

var_h2o = dat_new.createVariable("water_vapor","f4",("site","layer")); var_h2o[:] = q
var_co =  dat_new.createVariable("carbon_monoxide","f4",("site","layer")); var_co[:] = co
var_o3 =  dat_new.createVariable("ozone","f4",("site","layer")); var_o3[:] = o3
var_ch4 = dat_new.createVariable("methane","f4",("site","layer")); var_ch4[:] = ch4
var_co2 = dat_new.createVariable("carbon_dioxide","f4",("site","layer")); var_co2[:] = co2
var_no2 = dat_new.createVariable("nitrogen_dioxide","f4",("site","layer")); var_no2[:] = no2


# For following variables, get vertical profile from CKDMIP 
var_n2o = dat_new.createVariable("nitrous_oxide","f4",("layer")); 
var_cfc11 = dat_new.createVariable("cfc11","f4",("layer")); 
var_cfc12 = dat_new.createVariable("cfc12","f4",("layer")); 




# # --------------------------------------------------------
# # ------------ 2. SPECIFY RRTMGP K-DISTRIBUTION ----------
# # --------------------------------------------------------
# rte_rrtmgp_dir = '/home/peter/soft/rrtmgp-nn-training/'
# lw_kdist_file = "rrtmgp-data-lw-g256-2018-12-04.nc"
# sw_kdist_file = "rrtmgp-data-sw-g224-2018-12-04.nc"
# lw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + lw_kdist_file
# sw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + sw_kdist_file

# spectrum='longwave'
# if spectrum=='shortwave':
#     gas_kdist_path = sw_kdist_path
# else:
#     gas_kdist_path = lw_kdist_path

# kdist           = Dataset(gas_kdist_path)
# ngpt            = kdist.dimensions['gpt'].size
# # Temperature and pressure range of the LUT, for checking that the input
# # data does not exceed these
# kdist_temp_ref  = kdist.variables['temp_ref'][:]
# kdist_pres_ref  = kdist.variables['press_ref'][:]
# kdist_gases_raw = kdist.variables['gas_names'][:]
# kdist_gases     = ''.join(str(s, encoding='UTF-8') for s in kdist_gases_raw)
# kdist_gases     = kdist_gases.split()
# kdist.close()

# remaining RFMIP gases
# some already existed but are additionally provided as scalar "_GM" versions
varnames = ['methane_GM', 'carbon_dioxide_GM',
            'cfc11_GM', 'cfc12_GM','nitrous_oxide_GM',
            'carbon_monoxide_GM','hcfc22_GM','hfc23_GM','hfc32_GM',
            'hfc125_GM', 'hfc143a_GM','cf4_GM','hfc134a_GM',
            'carbon_tetrachloride_GM', 'oxygen_GM','nitrogen_GM']

for var in varnames:
    var_dat = dat_rfmip.variables[var][:].data
    #Get minimum value
    varmin = var_dat.min()
    varmax = var_dat.max()
    # create uniformly spaced numbers in the range
    vals = np.linspace(varmin,varmax,nexpt_new)
    # Shuffle
    np.random.shuffle(vals)
    # This is now one experiment. Save the data to the new file
    var_new = dat_new.createVariable(var,"f4",("expt"))
    var_new[:] = vals
    
    
varnames.extend(['water_vapor','carbon_monoxide_GM','ozone','methane_GM',
                 'pres_layer','pres_level','temp_level','temp_layer','surface_temperature','lon','lat'])  

varnames_new = varnames.copy()

for var in varnames:
    ncvar = dat_rfmip.variables[var]
    dat_new.variables[var].setncatts({k: ncvar.getncattr(k) for k in ncvar.ncattrs()})


# Create output files for RRTMGP: 
fpath_nninp = rootdir+'inp_lw_CAMS_1f1_NN2.nc'
fpath_nnoutp = rootdir+'tau_lw_CAMS_1f1_NN2.nc'

ngpt = 256
ngas = 21

dat_nninp = Dataset(fpath_nninp,'w',format='NETCDF4_CLASSIC')
dat_nninp.createDimension('level',nlay+1)
dat_nninp.createDimension('site',nsite)
dat_nninp.createDimension('expt',nexpt_new)
dat_nninp.createDimension('gas',ngas)
var_inp = dat_nninp.createVariable("col_gas","f4",("expt","site","level","gas"));

dat_nnoutp = Dataset(fpath_nnoutp,'w',format='NETCDF4_CLASSIC')
dat_nnoutp.createDimension('level',nlay+1)
dat_nnoutp.createDimension('site',nsite)
dat_nnoutp.createDimension('expt',nexpt_new)
dat_nnoutp.createDimension('gpt',ngpt)
var_outp = dat_nnoutp.createVariable("tau_lw","f4",("expt","site","level","gpt"));
var_outp2 = dat_nnoutp.createVariable("planck_frac","f4",("expt","site","level","gpt"));

dat_nninp.close()
dat_nnoutp.close()
dat.close()
dat_rfmip.close()
dat_new.close()