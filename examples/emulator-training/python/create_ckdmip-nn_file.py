#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:25:16 2021

@author: peter
"""



import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

# Load the data
rootdir = '/media/peter/samlinux/gdrive/phd/soft/python/nn-rrtmgp-new/'

fname = rootdir+'ckdmip_mmm_concentrations.nc'

fname_old = rootdir+'inputs_RFMIP.nc'
fname_new = rootdir+'inputs_CKDMIP-MMM-NEW.nc'

dat_ckdmip = Dataset(fname)
dat_rfmip = Dataset(fname_old)
dat_new   = Dataset(fname_new,'w')


nexpt_old = dat_rfmip.dimensions['expt'].size
nexpt_new = 1

p_lay = dat_ckdmip.variables['pressure_fl'][:].data
p_lev = dat_ckdmip.variables['pressure_hl'][:].data
nlay_old = dat_ckdmip.dimensions['level'].size
nlev_old = nlay_old + 1


# First step
# The lowest pressure of CKDMIP-MMM is not very low,
# so we will add one more layer, and increase the maximum temperature 
# (temp of this new layer) while we're at it
# It is easier to do this immediately so add one more level to the CKDMIP data
# and save this to a new file that we will then work with 
nlay = nlay_old+1
nlev = nlev_old+1

play_max = 109660.0
plev_max = 1.05 * play_max
tlay_max = 333.0
tlev_max = 336.0

temp_lay = dat_ckdmip.variables['temperature_fl'][:].data;  # (nlay,nsite)
temp_lev = dat_ckdmip.variables['temperature_hl'][:].data

h2o_med = dat_ckdmip.variables['h2o_median_mole_fraction_fl'][:].data 
h2o_min = dat_ckdmip.variables['h2o_minimum_mole_fraction_fl'][:].data 
h2o_max = dat_ckdmip.variables['h2o_maximum_mole_fraction_fl'][:].data 

o3_med = dat_ckdmip.variables['o3_median_mole_fraction_fl'][:].data 
o3_min = dat_ckdmip.variables['o3_minimum_mole_fraction_fl'][:].data 
o3_max = dat_ckdmip.variables['o3_maximum_mole_fraction_fl'][:].data 

# The existing "columns" are min,max,median of temperature, h2o and o3 
# = 3*3*3=27 combinations.
# Lets extend this dimension to min,max,median of CO2 and CH4 as well,
# which were previously separate variables. This results in 3**5 =  243 columns


nsite = 3**5

dat_new.createDimension('expt',None)
dat_new.createDimension('layer',nlay)
dat_new.createDimension('level',nlev)
dat_new.createDimension('site',nsite)

var_expt = dat_new.createVariable("expt_label","str",("expt"));
# Create first experiment so that the dimension is not empty
var_expt[0] = "Present-day N2O, CFC11-eq and CFC-12; gases not used in CKDMIP set to zero"

var_col = dat_new.createVariable("site_label","str",("site"));

# Column (row index for temp, h2o)
i = 0 
var_col[i] = "temperature-MED, H2O-MED, O3-MED"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MED, O3-MED"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MED, O3-MED"; i = i + 1
var_col[i] = "temperature-MED, H2O-MIN, O3-MED"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MIN  O3-MED"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MIN, O3-MED"; i = i + 1
var_col[i] = "temperature-MED, H2O-MAX, O3-MED"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MAX, O3-MED"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MAX, O3-MED"; i = i + 1

var_col[i] = "temperature-MED, H2O-MED, O3-MIN"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MED, O3-MIN"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MED, O3-MIN"; i = i + 1
var_col[i] = "temperature-MED, H2O-MIN, O3-MIN"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MIN, O3-MIN"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MIN, O3-MIN"; i = i + 1
var_col[i] = "temperature-MED, H2O-MAX, O3-MIN"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MAX, O3-MIN"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MAX, O3-MIN"; i = i + 1
var_col[i] = "temperature-MED, H2O-MED, O3-MAX"; i = i + 1

var_col[i] = "temperature-MIN, H2O-MED, O3-MAX"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MED, O3-MAX"; i = i + 1
var_col[i] = "temperature-MED, H2O-MIN, O3-MAX"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MIN, O3-MAX"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MIN, O3-MAX"; i = i + 1
var_col[i] = "temperature-MED, H2O-MAX, O3-MAX"; i = i + 1
var_col[i] = "temperature-MIN, H2O-MAX, O3-MAX"; i = i + 1
var_col[i] = "temperature-MAX, H2O-MAX, O3-MAX"; i = i + 1

# First lets recreate the table above, and then add CO2 and CH4 later as 
# additional variables to be varied, repeating the table three times upon each variable addition
# (the last variable to be added varies the slowest´)


# Create netCDF variables with site-dependency only
var_h2o         = dat_new.createVariable("water_vapor","f4",("site","layer"));
var_o3          = dat_new.createVariable("ozone",      "f4",("site","layer"));
var_co2         = dat_new.createVariable("carbon_dioxide", "f4",("site","layer"));
var_ch4         = dat_new.createVariable("methane",       "f4",("site","layer"));
var_p_lay       = dat_new.createVariable("pres_layer","f4",("layer"));
var_p_lev       = dat_new.createVariable("pres_level","f4",("level")); 
var_temp_lay    = dat_new.createVariable("temp_layer","f4",("site","layer")); 
var_temp_lev    = dat_new.createVariable("temp_level","f4",("site","level")); 
var_tempsfc     = dat_new.createVariable("surface_temperature","f4",("site"));
var_sfc_emis    = dat_new.createVariable("surface_emissivity","f4",("site")); 

# Water vapor
# First 9 columns
h2o = np.concatenate((h2o_med,h2o_min,h2o_max), axis=0)
# Tiled 3 times
h2o = np.tile(h2o,reps=(3,1))

# Ozone
# This variable is not temperature dependent so it doesn't matter 
# if we use e.g. o3_min[0] or o3_min[1]

o3_med = np.repeat(o3_med, repeats=3, axis=0)
o3_min = np.repeat(o3_min, repeats=3, axis=0)
o3_max = np.repeat(o3_max, repeats=3, axis=0)
o3 = np.concatenate((o3_med,o3_min,o3_max), axis=0)


# Pressure and temperature 
p_lay = np.tile(p_lay,reps=(9,1))
p_lev = np.tile(p_lev,reps=(9,1))
temp_lay = np.tile(temp_lay,reps=(9,1))
temp_lev = np.tile(temp_lev,reps=(9,1))


# --------- ADD CO2 -----------

# First lets tile the existing variables 3 times

var_col[0:(3*27)] = np.tile(var_col[0:27],3)

o3          = np.tile(o3,reps=(3,1))
h2o         = np.tile(h2o,reps=(3,1))
temp_lay    = np.tile(temp_lay,reps=(3,1))
temp_lev    = np.tile(temp_lev,reps=(3,1))
p_lay       = np.tile(p_lay,reps=(3,1))
p_lev       = np.tile(p_lev,reps=(3,1))

# Lets get the CO2 data
co2_orig = 1E6 * np.tile(dat_ckdmip.variables["co2_mole_fraction_fl"][:],reps=(9,1))
co2_fut  = (2274.54 / 415.0) * co2_orig
co2_LGM  = (180 / 415.0) * co2_orig # Same as co2_min
co2_mean = ((0.5*(2240-180))/415.0) * co2_orig

# concatenate the CO2 variables (first MEAN, then MIN, then MAX) to create the big co2 variable
co2 = np.concatenate((co2_mean,co2_LGM,co2_fut), axis=0)


# Add CO2 information to labels
for i in range(0,27):
    var_col[i] = var_col[i] + ", CO2-MEAN"
    
for i in range(1*27,2*27):
    var_col[i] = var_col[i] + ", CO2-MIN "

for i in range(2*27,3*27):
    var_col[i] = var_col[i] + ", CO2-MAX "



# --------- ADD CH4 -----------

# First lets tile the existing variables 3 times
var_col[0:(3*81)] = np.tile(var_col[0:81],3)

o3          = np.tile(o3,reps=(3,1))
h2o         = np.tile(h2o,reps=(3,1))
co2         = np.tile(co2,reps=(3,1))
temp_lay    = np.tile(temp_lay,reps=(3,1))
temp_lev    = np.tile(temp_lev,reps=(3,1))
p_lay       = np.tile(p_lay,reps=(3,1))
p_lev       = np.tile(p_lev,reps=(3,1))

# Lets get the CH4 data. This time we want 81 of each (original 3 repeated 27 times),
# and these 81-length mean,min,max again tiled after each other since its the slowest varying "dimension".
ch4_orig = 1E9 * np.tile(dat_ckdmip.variables["ch4_mole_fraction_fl"][:],reps=(27,1))
ch4_fut      = (3500 / 1921.0) * ch4_orig # Same as max
ch4_LGM      = (350 / 1921.0) * ch4_orig  # Same as min
ch4_mean = ((0.5*(3500-350))/1921.0) * ch4_orig

ch4 = np.concatenate((ch4_mean,ch4_LGM,ch4_fut), axis=0)

# Add CH4 information to labels
for i in range(0,81):
    var_col[i] = var_col[i] + ", CH4-MEAN"
    
for i in range(1*81,2*81):
    var_col[i] = var_col[i] + ", CH4-MIN"

for i in range(2*81,3*81):
    var_col[i] = var_col[i] + ", CH4-MAX"


# Write these site-variables
var_p_lay[:] = p_lay
var_p_lev[:] = p_lev
var_temp_lay[:] = temp_lay
var_temp_lev[:] = temp_lev
var_tempsfc[:] = temp_lev[:,-1]
var_sfc_emis[:] = 0.98

var_h2o[:] = h2o
var_o3[:]  = o3
var_co2[:]  = co2
var_ch4[:]  = ch4



# GASES 
# First lets create the NetCDF variables and copy their attributes from RFMIP


# Create remaining gases180
remaining_gases = ['carbon_monoxide_GM','carbon_tetrachloride_GM', 'cf4_GM',
 'cfc11_GM', 'cfc12_GM','hcfc22_GM','hfc125_GM','hfc134a_GM','hfc143a_GM','hfc23_GM',
 'hfc32_GM', 'nitrous_oxide_GM','nitrogen_GM','oxygen_GM']
 
for var in remaining_gases:
    ncvar = dat_rfmip.variables[var]
    var_new = dat_new.createVariable(var,"f4",("expt","layer")); var_new[:] = 0.0
 
    
    
# Copy attributes of all variables    
vars_all = ['temp_layer','temp_level', 'pres_layer', 'pres_level', 'expt_label',
            'surface_temperature','water_vapor', 'surface_emissivity',
            'carbon_dioxide_GM','carbon_monoxide_GM','carbon_tetrachloride_GM', 'cf4_GM',
            'cfc11_GM', 'cfc12_GM','hcfc22_GM','hfc125_GM','hfc134a_GM','hfc143a_GM','hfc23_GM',
            'hfc32_GM', 'methane_GM', 'nitrous_oxide_GM',"ozone",'nitrogen_GM','oxygen_GM']
    
for var in vars_all:
    ncvar = dat_rfmip.variables[var]
    dat_new.variables[var].setncatts({k: ncvar.getncattr(k) for k in ncvar.ncattrs()})


# - EXPERIMENTS:
# - 1 CKDMIP-exps
#     - P.D. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero

# - = 1 exp 243 columns
    
vars_gases = ['carbon_dioxide_GM','carbon_monoxide_GM','carbon_tetrachloride_GM', 'cf4_GM',
'cfc11_GM', 'cfc12_GM','hcfc22_GM','hfc125_GM','hfc134a_GM','hfc143a_GM','hfc23_GM',
'hfc32_GM', 'methane_GM', 'nitrous_oxide_GM'] 

vars_gases_ckdmip = ['carbon_dioxide_GM','cfc11_GM', 'cfc12_GM',
'methane_GM', 'nitrous_oxide_GM'] 
vars_gases_ckdmip_originalnames = ['co2_mole_fraction_fl','cfc11_mole_fraction_fl', 
'cfc12_mole_fraction_fl', 'ch4_mole_fraction_fl', 'n2o_mole_fraction_fl'] 

vars_gases_nockdmip = list(set(vars_gases) - set(vars_gases_ckdmip))
      

# The CKDMIP gas data using RFMIP units 
cfc11_orig = 1E12 * dat_ckdmip.variables["cfc11_mole_fraction_fl"][0,:]
cfc12_orig = 1E12 * dat_ckdmip.variables["cfc12_mole_fraction_fl"][0,:]
n2o_orig = 1E9 * dat_ckdmip.variables["n2o_mole_fraction_fl"][0,:],


# Create preindustrial, future and LGM (LastGlacialMaximum) profiles using
# Table 2 in paper: "R. J. Hogan and M. Matricardi: Correlated K-Distribution Model 
# Intercomparison Project (CKDMIP)"
# The profiles are multiplied by the ratio of the surface mole fraction values between the 
# experiment and present-day


cfc11_PI    = (32.0 / 861.0) * cfc11_orig # this is CFC11-eq for the experiments using only CKDMIP gases
cfc12_PI    = (0.0  / 495.0) * cfc12_orig
n2o_PI      = (270.0 / 332.0) * n2o_orig

cfc11_fut    = (2000 / 861.0) * cfc11_orig  # this is CFC11-eq for the experiments using only CKDMIP gases
cfc12_fut    = (200  / 495.0) * cfc12_orig
n2o_fut      = (405  / 332.0) * n2o_orig

cfc11_LGM    = (32 / 861.0) * cfc11_orig  # this is CFC11-eq for the experiments using only CKDMIP gases
cfc12_LGM    = (0  / 495.0) * cfc12_orig
n2o_LGM      = (190  / 332.0) * n2o_orig
n2o_max      = (540  / 332.0) * n2o_orig

# Oxygen and nitrogen are still missing
iexp_ref = 0  # doesn't matter, its constant
dat_new.variables["oxygen_GM"][:] = dat_rfmip.variables["oxygen_GM"][iexp_ref].data 
dat_new.variables["nitrogen_GM"][:] = dat_rfmip.variables["nitrogen_GM"][iexp_ref].data 

# For other gases max, min correspond to Future, LGM

# ----------- Exp 1:  P.D. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
iexp = 0
   
var_expt[iexp] = " P.D. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero"

# i = 0
# for varname in vars_gases_ckdmip:
#     print(varname)
#     varname_orig = vars_gases_ckdmip_originalnames[i]
#     var_dat = dat_ckdmip.variables[varname_orig][:]
#     var_dat = np.tile(var_dat,reps=(9,1))
#     dat_new.variables[varname][iexp,:] =  var_dat
#     i = i + 1

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_orig
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_orig
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_orig



# Sanity check
for v in dat_new.variables:
    print(v); 
    datt = dat_new.variables[v][:].data
    print("MAX ",np.max(datt),"   MIN ",np.min(datt))    
    
# Non_CKDMIP-gases were missing for exps 1-13, set to zero
for iexp in range(0,14):
    for varname in vars_gases_nockdmip:
       dat_new.variables[varname][iexp] = 0.0  
       

dat_new.close()

