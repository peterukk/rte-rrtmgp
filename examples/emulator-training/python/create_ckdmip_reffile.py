#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:55:24 2021

@author: peter
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:47:12 2020

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


fname_old = rootdir+'inputs_CKDMIP-MM-Big.nc'
dat = Dataset(fname_old)

fname_new = rootdir+'inputs_CKDMIP-MMM-extended.nc'
dat_new   = Dataset(fname_new,'w')


nsite = dat.dimensions['site'].size
nlay  = dat.dimensions['layer'].size
nlev  = dat.dimensions['level'].size


# The lowest pressure of CKDMIP-MMM is not very low,
# so we will add one more layer, and increase the maximum temperature 
# (temp of this new layer) while we're at it
nlay_new = nlay+1
nlev_new = nlev+1

play_max = 109660.0
plev_max = 1.05 * play_max
tlay_max = 333.0
tlev_max = 336.0
co2_max = 2274.54

dat_new.createDimension('expt',None)
dat_new.createDimension('layer',nlay_new)
dat_new.createDimension('level',nlev_new)
dat_new.createDimension('site',nsite)

var_expt = dat_new.createVariable("expt_label","str",("expt"));
# Create first experiment so that the dimension is not empty
var_expt[0] = "Present-day N2O, CFC11-eq and CFC-12; minor gases not used in CKDMIP set to zero"

var_col = dat_new.createVariable("site_label","str",("site"));

var_col =  dat.variables['site_label'][:]


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

var_cfc11       = dat_new.createVariable("cfc11", "f4",("expt","layer"));
var_cfc12       = dat_new.createVariable("cfc12", "f4",("expt","layer"));
var_n2o         = dat_new.createVariable("nitrous_oxide", "f4",("expt","layer"));


dat_h2o  = dat.variables['water_vapor'][:,:]
dat_o3   = dat.variables['ozone'][:,:]
dat_co2  = dat.variables['carbon_dioxide_GM'][:,:]
dat_ch4  = dat.variables['methane_GM'][:,:]
dat_play  = dat.variables['pres_layer'][0,:]
dat_plev  = dat.variables['pres_level'][0,:]
dat_tlay  = dat.variables['temp_layer'][:,:]
dat_tlev  = dat.variables['temp_level'][:,:]



# Create remaining gases
remaining_gases = ['cfc11', 'cfc12', 'nitrous_oxide', 
  'carbon_monoxide_GM',  'carbon_tetrachloride_GM', 'cf4_GM',
 'hcfc22_GM','hfc125_GM','hfc134a_GM','hfc143a_GM','hfc23_GM', 'hfc32_GM', 
 'nitrogen_dioxide_GM','nitrogen_GM','oxygen_GM']
 
for var in remaining_gases:
    var_new = dat_new.createVariable(var,"f4",("expt"))
    var_new[:] = 0.0
 
    
    
# Copy attributes of all variables    
vars_all = ['temp_layer','temp_level', 'pres_layer', 'pres_level', 'expt_label',
            'surface_temperature','water_vapor', 'surface_emissivity',
            'carbon_dioxide_GM','carbon_monoxide_GM','carbon_tetrachloride_GM', 'cf4_GM',
            'cfc11_GM', 'cfc12_GM','hcfc22_GM','hfc125_GM','hfc134a_GM','hfc143a_GM','hfc23_GM',
            'hfc32_GM', 'methane_GM', 'nitrous_oxide_GM',"ozone",'nitrogen_GM','oxygen_GM']
    
for var in vars_all:
    ncvar = dat.variables[var]
    dat_new.variables[var].setncatts({k: ncvar.getncattr(k) for k in ncvar.ncattrs()})



# - EXPERIMENTS:
# - 14 CKDMIP-exps
#     - P.D. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
#     - P.I. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
#     - future N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
#     - LGM N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
#     - P.D. N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero
#     - P.D. N2O, min CFC11-eq, min CFC12, non-CKDMIP gases zero
#     - P.D. N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero
#     - P.D. N2O, max CFC11-eq, max CFC12, non-CKDMIP gases zero
#     - max N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero
#     - max N2O, min CFC11-eq, min CFC12, non-CKDMIP gases zero
#     - max N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero
#     - P.I. N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero
#     - P.I. N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero
#     - P.I. N2O, max CFC11-eq, max CFC12, non-CKDMIP gases zero    

    
# - 13 RFMIP-style exps: uses also remaining gases hfc23, hfc134a, hcfc22, 
#       hfc143a, cf4, hfc125, carbon_monoxide, hfc32, carbon_tetrachloride
#   Vertically constant profiles 
#     - P.D., non-CKDMIP gases vertically constant  
#     - P.I., non-CKDMIP gases vertically constant  
#     - future., non-CKDMIP gases vertically constant  
#     - P.D., P.I. N2O, non-CKDMIP gases zero
#     - P.D., P.I. carbon_monoxide_GM
#     - P.D., P.I. carbon_tetrachloride_GM
#     - P.D., P.I cf4_GM
#     - P.D., P.I hcfc22_GM
#     - P.D., P.I hfc125_GM
#     - P.D., P.I hfc134a_GM
#     - P.D., P.I. hfc143a_GM
#     - P.D., P.I. hfc23_GM
#     - P.D., P.I. hfc32_GM

# - = 24 exps, 243 columns
    
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

# ----------- Exp 2:  P.I. N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
iexp = 1
var_expt[iexp] = "P.I. (pre-industrial) N2O, CFC11-eq, CFC12, non-CKDMIP gases zero"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_PI
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_PI
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_PI

# ----------- Exp 3:  future. CH4, CO2, N2O, CFC11-eq, CFC12, non-CKDMIP gases zero
iexp = 2
var_expt[iexp] = "Future N2O, CFC11-eq, CFC12, non-CKDMIP gases zero"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_fut
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_fut

#  -----------  
iexp = 3
var_expt[iexp] = "LGM (Last Glacial Maximum) N2O, CFC11-eq, CFC12, non-CKDMIP gases zero"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_LGM
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_LGM
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_LGM



#  ----------- exp 5-8 (inds 4-7): P.D. N2O
for iexp in range(4,8):
    dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_orig
    
iexp = 4
var_expt[iexp] = "P.D. N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = 0.0
dat_new.variables["cfc12_GM"][iexp,:]         = cfc12_fut


#  ----------- 
iexp = 5
var_expt[iexp] = "P.D. N2O, min CFC11-eq, min CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = 0.0
dat_new.variables["cfc12_GM"][iexp,:]         = 0.0

#  ----------- 
iexp = 6
var_expt[iexp] = "P.D. N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:]         = 0.0

#  -----------
iexp = 7
var_expt[iexp] = "P.D. N2O, max CFC11-eq, max CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:]         = cfc12_fut


#  ----------- exp 9-11 (inds 8-10): max N2O
for iexp in range(8,11):
    dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_max
    
    
#  -----------
iexp = 8
var_expt[iexp] = "max N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = 0.0
dat_new.variables["cfc12_GM"][iexp,:]         = cfc12_fut

#  -----------
iexp = 9
var_expt[iexp] = "max N2O, min CFC11-eq, min CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = 0.0
dat_new.variables["cfc12_GM"][iexp,:]         = 0.0

#  -----------
iexp = 10
var_expt[iexp] = "max N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:]         = 0.0


#  ----------- exp 12-14 (inds 11-13): min  N2O
for iexp in range(11,14):
    dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_PI
    
#  -----------
iexp = 11
var_expt[iexp] = "min N2O, min CFC11-eq, max CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = 0.0
dat_new.variables["cfc12_GM"][iexp,:]         = cfc12_fut

#  -----------
iexp = 12
var_expt[iexp] = "min N2O, max CFC11-eq, min CFC12, non-CKDMIP gases zero"
dat_new.variables["cfc11_GM"][iexp,:]         = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:]         = 0.0

#  -----------
iexp = 13
var_expt[iexp] = "min N2O, max CFC11-eq, max CFC12, non-CKDMIP gases zero    "
dat_new.variables["cfc11_GM"][iexp,:]         = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:]         = cfc12_fut


# - 13 RFMIP-style exps: uses also remaining gases hfc23, hfc134a, hcfc22, 
#       hfc143a, cf4, hfc125, carbon_monoxide, hfc32, carbon_tetrachloride
#   Vertically constant profiles :

#     - P.D., non-CKDMIP gases vertically constant  
#     - P.I., non-CKDMIP gases vertically constant  
#     - future., non-CKDMIP gases vertically constant  
#     - P.D., P.I. N2O, non-CKDMIP...
#     - P.D., P.I. carbon_monoxide_GM
#     - P.D., P.I. carbon_tetrachloride_GM
#     - P.D., P.I cf4_GM
#     - P.D., P.I hcfc22_GM
#     - P.D., P.I hfc125_GM
#     - P.D., P.I hfc134a_GM
#     - P.D., P.I. hfc143a_GM
#     - P.D., P.I. hfc23_GM
#     - P.D., P.I. hfc32_GM


expt_label_rfmip = dat_rfmip["expt_label"][:]
#  ----------- 
iexp = 14
iexp_ref = 0 # The experiment index in RFMIP
var_expt[iexp] = "P.D., remaining RRTMGP gases vertically constant"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_orig
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_orig
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_orig

for varname in vars_gases_nockdmip:
    dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data 


#     - P.I., non-CKDMIP gases vertically constant  
#  ----------- 
iexp = 15
iexp_ref = 1 # The experiment index in RFMIP
var_expt[iexp] = "P.I., remaining RRTMGP gases vertically constant"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_PI
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_PI
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_PI

for varname in vars_gases_nockdmip:
    dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data 

#     - future., non-CKDMIP gases vertically constant  
iexp = 16
iexp_ref = 3 # The experiment index in RFMIP
var_expt[iexp] = "Future, remaining RRTMGP gases vertically constant"

dat_new.variables["cfc11_GM"][iexp,:] = cfc11_fut
dat_new.variables["cfc12_GM"][iexp,:] = cfc12_fut
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_fut

for varname in vars_gases_nockdmip:
    dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

    
#  ----------- ----------- 
    
# iexp 17-26 : P.D. except for select gases
iexp_ref = 0 # The experiment index in RFMIP

for iexp in range(17,27):
    dat_new.variables["cfc11_GM"][iexp,:] = cfc11_orig
    dat_new.variables["cfc12_GM"][iexp,:] = cfc12_orig
    dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_orig
    
    for varname in vars_gases_nockdmip:
        dat_new.variables[varname][iexp] = dat_rfmip.variables[varname][iexp_ref].data 
    
#  -----------     
iexp = 17
iexp_ref = 1 # The experiment index in RFMIP
var_expt[iexp] = "P.D., P.I. nitrous_oxide, non-CKDMIP gases vertically constant"
dat_new.variables["nitrous_oxide_GM"][iexp,:] = n2o_PI

#  ----------- 
iexp = 18; varname = "carbon_monoxide_GM"
var_expt[iexp] = "P.D., P.I. carbon_monoxide, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 19; varname = "carbon_tetrachloride_GM"
var_expt[iexp] = "P.D., P.I. carbon_tetrachloride, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 20; varname = "cf4_GM"
var_expt[iexp] = "P.D., P.I. cf4, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 21; varname = "hcfc22_GM"
var_expt[iexp] = "P.D., P.I. hcfc22, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 22; varname = "hfc125_GM"
var_expt[iexp] = "P.D., P.I. hfc125, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 23; varname = "hfc134a_GM"
var_expt[iexp] = "P.D., P.I. hfc134a, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data    
 
#  ----------- 
iexp = 24; varname = "hfc143a_GM"
var_expt[iexp] = "P.D., P.I. hfc143a, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data     

#  ----------- 
iexp = 25; varname = "hfc23_GM"
var_expt[iexp] = "P.D., P.I. hfc23, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data   

#  ----------- 
iexp = 26; varname = "hfc32_GM"
var_expt[iexp] = "P.D., P.I. hfc32, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = dat_rfmip.variables[varname][iexp_ref].data   



# Oxygen and nitrogen are still missing
iexp_ref = 0  # doesn't matter, its constant
dat_new.variables["oxygen_GM"][:] = dat_rfmip.variables["oxygen_GM"][iexp_ref].data 
dat_new.variables["nitrogen_GM"][:] = dat_rfmip.variables["nitrogen_GM"][iexp_ref].data 


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


# Create some new exps

import os
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

# Load the data
rootdir = '/home/peter/gdrive/phd/'

# fname = rootdir+'rte-rrtmgp-nn-dev/examples/rfmip-clear-sky/ckdmip_mmm_concentrations.nc'

# fname_old = rootdir+'rte-rrtmgp-nn-dev/examples/rfmip-clear-sky/inputs_RFMIP.nc'
fname_new = rootdir+'rte-rrtmgp-nn-dev/examples/rfmip-clear-sky/inputs_CKDMIP-MM-Big.nc'

# dat_ckdmip = Dataset(fname)
# dat_rfmip = Dataset(fname_old)
dat_new   = Dataset(fname_new,'a')

nsite = dat_new.dimensions['site'].size 
nlay = dat_new.dimensions['layer'].size 
nlev = dat_new.dimensions['level'].size 
nexpt = dat_new.dimensions['expt'].size 

expt_label = dat_new.variables["expt_label"][:]

for i in range(nexpt):
    print(i, expt_label[i], dat_new.variables['cfc12_GM'][i,-1])
    
dat_new.variables["expt_label"][17] =  "P.I. nitrous_oxide, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][18] =  "P.I. carbon_monoxide, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][19] =  "P.I. carbon_tetrachloride, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][20] =  "P.I. cf4, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][21] =  "P.I. hcfc22, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][22] =  "P.I. hfc125, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][23] =  "P.I. hfc134a, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][24] =  "P.I. hfc143a, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][25] =  "P.I. hfc23, remaining RRTMGP gases vertically constant"
dat_new.variables["expt_label"][26] =  "P.I. hfc32, remaining RRTMGP gases vertically constant"



vars_gases_ckdmip = ['oxygen_GM','nitrogen_GM','carbon_dioxide_GM','cfc11_GM', 'cfc12_GM',
'methane_GM', 'nitrous_oxide_GM'] 

varlist_expt = []
for v in dat_new.variables:
    if (dat_new.variables[v].dimensions[0] == 'expt'):
        varlist_expt.append(v)
        
varlist_expt.remove("expt_label")        
# gases not on CKDMIP
varlist_expt_nockdmip = list(set(varlist_expt) - set(vars_gases_ckdmip))



iexp_ref = 0
for iexp in range(27,34+1):
    for varname in varlist_expt:
        dat_new.variables[varname][iexp] = dat_new.variables[varname][iexp_ref]

    
iexp = 27
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 0, non-CKDMIP gases zero"
varname = "cfc11_GM"
fac = 0/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 
    
    
iexp = 28
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 200, non-CKDMIP gases zero"
varname = "cfc11_GM"
fac = 200/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 


iexp = 29
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 800, non-CKDMIP gases zero"
fac = 800/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 


iexp = 30
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 1400, non-CKDMIP gases zero"
fac = 1400/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 

iexp = 31
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 2000, non-CKDMIP gases zero"
fac = 2000/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 


iexp = 32
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 200, non-CKDMIP gases zero"
varname = "cfc12_GM"
fac = 200/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 

iexp = 33
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 400, non-CKDMIP gases zero"
fac = 400/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 

iexp = 34
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 520, non-CKDMIP gases zero"
fac = 520/dat_new.variables[varname][iexp][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][iexp] 


iexp_ref = 1
for iexp in range(35,38+1):
    for varname in varlist_expt:
        dat_new.variables[varname][iexp] = dat_new.variables[varname][iexp_ref]
        

iexp = 35
dat_new.variables["expt_label"][iexp] = "P.I., CFC11 800, non-CKDMIP gases zero"
varname = "cfc11_GM"
fac = 800/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 36
dat_new.variables["expt_label"][iexp] = "P.I., CFC11 2000, non-CKDMIP gases zero"
fac = 2000/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 


iexp = 37
dat_new.variables["expt_label"][iexp] = "P.I., CFC12 200, non-CKDMIP gases zero"
varname = "cfc12_GM"
fac = 200/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 38
dat_new.variables["expt_label"][iexp] = "P.I., CFC12 600, non-CKDMIP gases zero"
fac = 600/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 




iexp_ref = 14
for iexp in range(39,43+1):
    for varname in varlist_expt:
        dat_new.variables[varname][iexp] = dat_new.variables[varname][iexp_ref]
        

iexp = 39
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 0, remaining RRTMGP gases vertically constant"
varname = "cfc11_GM"
fac = 0/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 40
dat_new.variables["expt_label"][iexp] = "P.D., CFC11 250, remaining RRTMGP gases vertically constant"
fac = 250/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 41
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 200, remaining RRTMGP gases vertically constant"
varname = "cfc12_GM"
fac = 200/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 42
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 400, remaining RRTMGP gases vertically constant"
fac = 400/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 43
dat_new.variables["expt_label"][iexp] = "P.D., CFC12 600, remaining RRTMGP gases vertically constant"
fac = 600/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 



iexp_ref = 16
for iexp in range(43,47+1):
    for varname in varlist_expt:
        dat_new.variables[varname][iexp] = dat_new.variables[varname][iexp_ref]
        

iexp = 43
dat_new.variables["expt_label"][iexp] = "future, CFC11 0, remaining RRTMGP gases vertically constant"
varname = "cfc11_GM"
fac = 0/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 44
dat_new.variables["expt_label"][iexp] = "future, CFC11 250, remaining RRTMGP gases vertically constant"
fac = 250/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 45
dat_new.variables["expt_label"][iexp] = "future, CFC12 200, remaining RRTMGP gases vertically constant"
varname = "cfc12_GM"
fac = 200/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 46
dat_new.variables["expt_label"][iexp] = "future, CFC12 400, remaining RRTMGP gases vertically constant"
fac = 400/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 

iexp = 47
dat_new.variables["expt_label"][iexp] = "future, CFC12 520, remaining RRTMGP gases vertically constant"
fac = 520/dat_new.variables[varname][0][-1]
dat_new.variables[varname][iexp] = fac * dat_new.variables[varname][0] 


#Sanity check
for v in dat_new.variables:
    print(v); 
    datt = dat_new.variables[v][:].data
    print("MAX ",np.max(datt),"   MIN ",np.min(datt))    
    

dat_new.close()


# Fix CFC11 being CFC11-eq when all gases are included

for iexp in range(15,26+1):
    fac = 0/dat_new.variables["cfc11_GM"][iexp][-1]
    dat_new.variables["cfc11_GM"][iexp] = fac * dat_new.variables["cfc11_GM"][iexp] 
    

for iexp in range(41,42+1):
    fac = 234/dat_new.variables["cfc11_GM"][iexp][-1]
    dat_new.variables["cfc11_GM"][iexp] = fac * dat_new.variables["cfc11_GM"][iexp] 
    
for iexp in range(45,47+1):
    fac = 57/dat_new.variables["cfc11_GM"][iexp][-1]
    dat_new.variables["cfc11_GM"][iexp] = fac * dat_new.variables["cfc11_GM"][iexp] 
    
    
# Again some more exps
    
    
#  ----------- ----------- 
    
# iexp 48-58 : P.I. except for select gases which are half of value
    
iexp_ref = 15 # The experiment index in RFMIP

for iexp in range(48,57+1):
    for varname in varlist_expt:
        dat_new.variables[varname][iexp] = dat_new.variables[varname][iexp_ref].data 
    
#  -----------     
       
#  -----------     
iexp = 48; varname = "nitrous_oxide_GM"
var_expt[iexp] = "P.I., 0.6x max nitrous_oxide, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][8,:]

#  ----------- 
iexp = 49; varname = "carbon_monoxide_GM"
var_expt[iexp] = "P.I., 0.6x max carbon_monoxide, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 50; varname = "carbon_tetrachloride_GM"
var_expt[iexp] = "P.I., 0.6x max carbon_tetrachloride, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 51; varname = "cf4_GM"
var_expt[iexp] = "P.I., 0.6x max cf4, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 52; varname = "hcfc22_GM"
var_expt[iexp] = "P.I., 0.6x max hcfc22, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 53; varname = "hfc125_GM"
var_expt[iexp] = "P.I., 0.6x max hfc125, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 54; varname = "hfc134a_GM"
var_expt[iexp] = "P.I., 0.6x max hfc134a, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()
 
#  ----------- 
iexp = 55; varname = "hfc143a_GM"
var_expt[iexp] = "P.I., 0.6x max hfc143a, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 56; varname = "hfc23_GM"
var_expt[iexp] = "P.I., 0.6x max hfc23, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()

#  ----------- 
iexp = 57; varname = "hfc32_GM"
var_expt[iexp] = "P.I., 0.6x max hfc32, non-CKDMIP gases vertically constant"
dat_new.variables[varname][iexp,:] = 0.6 * dat_new.variables[varname][:].max()
