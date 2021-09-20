#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:43:11 2019

@author: Peter Ukkonen
"""

import os
from netCDF4 import Dataset,num2date
import numpy as np
this_dir    = os.getcwd() + "/"
os.chdir(this_dir)
root_dir    = this_dir + "../../" # emulator-training directory
# root_dir = '/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/'
from hum import esat, mixr2rh

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def gfsCloudFraction(p,qc,rh,T):
 # Code I found for computing cloud fraction
 # https://atmos.uw.edu/~cjones/nobackup/gfsCloudFraction.html
 # GFSCLOUDFRACTION diagnoses cloud fraction as in GFS
 # Inputs:
 #    p:  pressure [in Pa]
 #    qc: cloud condensate mixing ratio [kg/kg]
 #    rh: relative humidity [unitless fraction]
 #    T:  temperature [K]
 # Output:
 #   cloudFraction: cloud fraction [unitless fraction]
 # Notes:
 #    (1) This script depends on ../thermo/ function qs
    
    cloudWaterThreshold = 1e-6 * p *1e-5 # original has p in hPa
    
    isCloud = (qc>cloudWaterThreshold);
    qsat = qsatGFS(p,T)
    
    tem1 = ((1-rh[isCloud]) * qsat[isCloud])**(0.49) #suggests rh \in [0,1]
    tem1 = 100 / (np.minimum(np.maximum(tem1,1e-4),1)); # bounded by 100 <= tem1 <= 1e6;
    
    val = np.maximum(tem1*qc[isCloud], 0); # no need to apply threshold of val<50
    
    tem2 = rh[isCloud]**(1/4)
    cloudFraction = np.zeros(np.shape(qc))
    cloudFraction[isCloud] = np.maximum(tem2*(1-np.exp(-val)),0)
    return cloudFraction

def qsatGFS(p,T):
    # QSATGFS returns saturation humidity for give pressure (p) and
    # temperature (T).
   
    # NOTE: this is saturation _humidity_, not _mixing ratio_:
    #   using q = humidity and w = mixing ratio, q = w/(1+w)
    
    Rd = 2.8705e2 # gas constant of dry air [J/kg/K]
    Rv = 4.6150e2 # gas constant water [J/kg/K]
    
    ep = Rd/Rv
    esatt = esat(T)
    #esat = esGFS(T);
    
    esatt = np.minimum(esatt,p)
    qs = ep*esatt / (p+(ep-1)*esatt)
    return qs 

# Input data file
fpath_cams  = root_dir+'data_input/CAMS_2018.nc'
# New file
fpath_new   = os.path.splitext(fpath_cams)[0] + '_RFMIPstyle2.nc'
print("Saving new file to {}".format(fpath_new))
# RFMIP file, used to copy attributes and compare
fpath_rfmip = root_dir+'data_input/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc'

dat         = Dataset(fpath_cams)
dat_new     = Dataset(fpath_new,'w')
dat_rfmip   = Dataset(fpath_rfmip)

lon         = dat.variables['lon'][:].data; nlon = lon.size
lat         = dat.variables['lat'][:].data; nlat = lat.size
nlay        = dat.variables['lev'][:].data.size
ntime       = dat.variables['time'][:].data.size

nsite = lon.size * lat.size * ntime
nexpt_new = 1

# create dimensions
dat_new.createDimension('layer',nlay)
dat_new.createDimension('level',nlay+1)
dat_new.createDimension('nhym',nlay)
dat_new.createDimension('site',nsite)
dat_new.createDimension('expt',nexpt_new)

# Write dimension variables

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
timedat = dat.variables['time'][:]
timedatt = timedat.reshape(1,ntime).repeat(nlat*nlon,axis=0)
timedatt = timedatt.reshape(nlon*nlat*ntime)
timevar[:] = timedatt

# Level
layvar = dat_new.createVariable("lev","f4",("layer"))
lev_v = dat.variables['lev']
layvar[:] = lev_v[:]

hyam = dat_new.createVariable("hyam","f4",("nhym"))
hybm = dat_new.createVariable("hybm","f4",("nhym"))
hyam[:] = dat.variables['hyam'][:]
hybm[:] = dat.variables['hybm'][:]

# save solar zenith angle
from sunposition import sunpos
t_unit =  dat.variables['time'].units
t_cal  =  dat.variables['time'].calendar
times = num2date(timedatt,units = t_unit,calendar = t_cal).data
az,zen = sunpos(times.data,latt,lonn,0)[:2] #discard RA, dec, H

szavar = dat_new.createVariable("solar_zenith_angle","f4",("site"))
szavar[:] = zen

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
                                 
for i in range(nsite):
    p_lev[i,1:-1] = moving_average(p_lay[i,:],2)
    p_lev[i,nlay] = 0.5*(p_lay[i,nlay-1] + sp[i])
    temp_lev[i,1:-1] = moving_average(temp_lay[i,:],2)
    temp_lev[i,nlay] = 0.5*(temp_lay[i,nlay-1] + temp_sfc[i])    

temp_lev[:,0] = temp_lay[:,0] + (p_lev[:,0] - p_lay[:,0]) * (temp_lay[:,1]-temp_lay[:,0]) / (p_lay[:,1]-p_lay[:,0])


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

#surface temperature and emissivity
var_tempsfc = dat_new.createVariable("surface_temperature","f4",("site"));
var_tempsfc[:] = temp_sfc
var_sfc_emis = dat_new.createVariable("surface_emissivity","f4",("site")); 
var_sfc_emis[:] = 0.5

#surface albedo
var_albedo = dat_new.createVariable("surface_albedo","f4",("site"));
var_albedo[:] = vars_reshaped['fal'][:]

# cloud variables
var_ciwc = dat_new.createVariable("ciwc","f4",("site","layer")); 
var_clwc = dat_new.createVariable("clwc","f4",("site","layer")); 
var_ciwc[:] = vars_reshaped['ciwc'][:]
var_clwc[:] = vars_reshaped['clwc'][:]

# solar irradiance in Joule
var_tisr = dat_new.createVariable("tisr","f4",("site"));
var_tisr[:] = vars_reshaped['tisr'][:]

# # solar zenith angle: random between 0-90
# var_sza = dat_new.createVariable("sza_rnd","f4",("site"));
# var_sza[:] =  np.random.random_sample(size=nsite)
# var_sza.long_name = "TOA incident solar radiation random value 0-90"
# var_sza.standard_name = "solar_zenith_angle"
# var_sza.units = "degree"

# COMPUTE CLOUD FRACTION
# First compute relative hum
q_lay = vars_reshaped['q'].data # mixing ratio
q_lay[q_lay<0] = 0.0
rh_lay = mixr2rh(q_lay,p_lay,temp_lay)

# cloud condensate mixing ratio: is this ice and water content combined?
qc = vars_reshaped['ciwc'][:] + vars_reshaped['clwc'][:]
cloudfrac =  gfsCloudFraction(p_lay,qc,rh_lay,temp_lay)
cloudfrac[cloudfrac>1.0] = 1.0 # some were slightly above 1
# mean cloud fraction in ifs ecrad_meridian  4.7168146965032728E-002
# here its just 1.9e-002

# As sanity check, estimate cloud mask for columns 
# If cloud frac is > 0.05 at any level, the mask will be 1
# cloudmask = np.zeros(nsite)
# for i in range(nsite):
#     cloudmask[i] = np.any(cloudfrac[i,:]>0.05)
# cloudmask.mean()
# Out[63]: 0.7027083333333334
# Sounds about right
    
var_cloudfrac =  dat_new.createVariable("cloud_fraction","f4",("site","layer")); 
var_cloudfrac[:] = cloudfrac
var_cloudfrac.long_name = 'Cloud fraction'
var_cloudfrac.units = '1'
comment = 'computed using function gfsCloudFraction, see ' \
  'https://atmos.uw.edu/~cjones/nobackup/gfsCloudFraction.html'
var_cloudfrac.comment = comment


#EXISTING GASES
# (site,layer                            FROM mass mixing ratio to mole fraction
ch4 = vars_reshaped['ch4'].data         * 28.9644 / 16.0425   * 1/float(dat_rfmip.variables['methane_GM'].units)
o3  = vars_reshaped['go3'].data         * 28.9644 / 47.9982   * 1/float(dat_rfmip.variables['ozone'].units)
co2 = vars_reshaped['co2'].data         * 28.9644 / 44.0095    * 1/float(dat_rfmip.variables['carbon_dioxide_GM'].units)
co  = vars_reshaped['co'].data          * 28.9644 / 28.0101   * 1/float(dat_rfmip.variables['carbon_monoxide_GM'].units)
no2 = vars_reshaped['no2'].data         * 28.9644 / 46.0055   * 1/1e-6
q   = vars_reshaped['q'].data           * 28.9644 / 18.01528  * 1/float(dat_rfmip.variables['water_vapor'].units)
# Nitrous oxide, downloaded from another CAMS data set, is already in mole fraction
# and in the correct units
n2o = vars_reshaped['N2O'].data

ch4 = np.reshape(ch4,(nsite,nlay))
o3  = np.reshape(o3,(nsite,nlay))
co2 = np.reshape(co2,(nsite,nlay))
co  = np.reshape(co,(nsite,nlay))
no2 = np.reshape(no2,(nsite,nlay))
q   = np.reshape(q,(nsite,nlay)); q[q<0] = 0.0
n2o = np.reshape(n2o,(nsite,nlay))

var_h2o = dat_new.createVariable("water_vapor","f4",("site","layer")); var_h2o[:] = q
var_o3 =  dat_new.createVariable("ozone","f4",("site","layer")); var_o3[:] = o3
var_ch4 = dat_new.createVariable("methane","f4",("site","layer")); var_ch4[:] = ch4
var_co2 = dat_new.createVariable("carbon_dioxide","f4",("site","layer")); var_co2[:] = co2
var_co =  dat_new.createVariable("carbon_monoxide","f4",("site","layer")); var_co[:] = co
var_no2 = dat_new.createVariable("nitrogen_dioxide","f4",("site","layer")); var_no2[:] = no2
var_n2o = dat_new.createVariable("nitrous_oxide","f4",("site","layer")); var_n2o[:] = n2o
# Oxygen and nitrogen (constants)
var_o2      = dat_new.createVariable("oxygen_GM","f4",("expt"))
var_o2[:]   = dat_rfmip.variables['oxygen_GM'][0]
var_n2      = dat_new.createVariable("nitrogen_GM","f4",("expt"))
var_n2[:]   = dat_rfmip.variables['nitrogen_GM'][0]

# ------------- WRITE ATTRIBUTES ------------
# Attributes for some variables are copied from the original file
for varname in ['lon','lat','time','lev','hyam','hybm','tisr','clwc','ciwc']:
    varin = dat.variables[varname]
    outVar = dat_new.variables[varname]
    print(varname)    
    # Copy variable attributes
    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})


# Attributes  for some variables are copied from RFMIP
varnames_copy = ['water_vapor','ozone','pres_layer','pres_level','temp_layer',
                 'temp_level','surface_temperature','surface_emissivity',
                 'oxygen_GM','nitrogen_GM', 'surface_albedo']

for var in varnames_copy:
    ncvar = dat_rfmip.variables[var]
    dat_new.variables[var].setncatts({k: ncvar.getncattr(k) for k in ncvar.ncattrs()})

# For other mutual vars, copy some attributes only
varnames_new = ['methane', 'carbon_dioxide', 'carbon_monoxide', 'nitrous_oxide']
varnames_rfmip = ['methane_GM', 'carbon_dioxide_GM', 'carbon_monoxide_GM', 'nitrous_oxide_GM']
attrs = ['units', 'standard_name']

i = 0
for var in varnames_new:
    ncvar = dat_rfmip.variables[varnames_rfmip[i]]    
    dat_new.variables[var].setncatts({k: ncvar.getncattr(k) for k in attrs})
    dat_new.variables[var].coordinates = "lon lat time" 
    i = i + 1

var_ch4.long_name = 'CH4 mole fraction'; var_ch4.original_name = 'CH4'
var_co2.long_name = 'CO2 mole fraction'; var_co2.original_name = 'CO2'
var_co.long_name = 'CO mole fraction';   var_co.original_name = 'CO'
var_n2o.long_name = 'N2O mole fraction'; var_n2o.original_name = 'N2O'
# nitrogen dioxide, not included in RFMIP
var_no2.long_name = 'NO2 mole fraction'; var_no2.original_name = 'NO2'
var_no2.standard_name = 'mole_fraction_of_nitrogen_dioxide_in_air'; var_no2.original_name = 'NO2'
var_no2.units = "1e-06"

# # For following variables, get vertical profile from CKDMIP 
# var_cfc11 = dat_new.createVariable("cfc11","f4",("layer")); 
# var_cfc12 = dat_new.createVariable("cfc12","f4",("layer")); 

var_expt = dat_new.createVariable("expt_label","str",("expt"));
var_expt[0] = "Present-day global CAMS profiles, no modification, RFMIP values for O2 N2"


# Sanity check
for v in dat_new.variables:
    print(v); 
    datt = dat_new.variables[v][:]
    print("MIN ",np.min(datt),"   MAX ",np.max(datt))  
    

dat.close()
dat_rfmip.close()
dat_new.close()
