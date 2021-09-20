#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:54:12 2021

@author: peter
"""

# Squash time and site into one dimension nsite_new = time*site so that
# site is the faster varying dimension (like in the original data),
# and then save a fraction e.g. 0.1 = 10% of these columns (randomly)
# to a new file

import os
from netCDF4 import Dataset,num2date
import numpy as np




# The proportion of columns to keep
frac = 0.2



# Input data file
fpath_cams  = "/media/peter/samsung/data/CAMS/CAMS_2015_RFMIPstyle.nc"
# New file
fpath_new   = os.path.splitext(fpath_cams)[0] + '_rndm_subset.nc'
print("Saving new file to {}".format(fpath_new))

dat         = Dataset(fpath_cams)
dat_new     = Dataset(fpath_new,'w')


nlay        = dat.variables['lev'][:].data.size
ntime       = dat.variables['time'][:].data.size
nsite       = dat.dimensions['site'].size 
nexpt       = 1
nlev = nlay + 1

nsite_new       = ntime * nsite
nsite_new_fin   = np.int(frac*nsite_new)



inds_keep  = np.random.default_rng().choice(nsite_new, nsite_new_fin, replace=False)
inds_keep.sort()



# create dimensions
dat_new.createDimension('layer',nlay)
dat_new.createDimension('level',nlay+1)
dat_new.createDimension('nhym',nlay)
dat_new.createDimension('site',nsite_new_fin)
dat_new.createDimension('expt',nexpt)
dat_new.createDimension('nv',3)


# Longitude latitude information into 1D (site) array
lonvar  = dat_new.createVariable("clon","f4",("site"))
latvar  = dat_new.createVariable("clat","f4",("site"))

lonbndvar  = dat_new.createVariable("clon_bnds","f4",("site","nv"))
latbndvar  = dat_new.createVariable("clat_bnds","f4",("site","nv"))

latbnd  = dat.variables['clon_bnds'][:].data
lonbnd  = dat.variables['clat_bnds'][:].data
# repeat from (nsite,3) to (nsite_new,3)
lonbnd = lonbnd.reshape(1,nsite,3).repeat(ntime,axis=0)
latbnd = latbnd.reshape(1,nsite,3).repeat(ntime,axis=0)
lonbnd = lonbnd.reshape(nsite_new,3); latbnd = latbnd.reshape(nsite_new,3)

lonn = dat.variables['clon'][:]
latt = dat.variables['clat'][:]
lonn = lonn.reshape(1, nsite).repeat(ntime,axis=0)
latt = latt.reshape(1, nsite).repeat(ntime,axis=0)
lonn = lonn.reshape(nsite_new); latt = latt.reshape(nsite_new)

lonbndvar[:] = latbnd[inds_keep,:]
latbndvar[:] = lonbnd[inds_keep,:]
lonvar[:] = lonn[inds_keep]
latvar[:] = latt[inds_keep]

# Time 
timevar = dat_new.createVariable("time","f4",("site"))
timedat = dat.variables['time'][:]
timedatt = timedat.reshape(ntime, 1).repeat(nsite,axis=1)
timedatt = timedatt.reshape(nsite*ntime)
timevar[:] = timedatt[inds_keep]

# Level
layvar = dat_new.createVariable("lev","f4",("layer"))
lev_v = dat.variables['lev']
layvar[:] = lev_v[:]

hyam = dat_new.createVariable("hyam","f4",("nhym"))
hybm = dat_new.createVariable("hybm","f4",("nhym"))
hyam[:] = dat.variables['hyam'][:]
hybm[:] = dat.variables['hybm'][:]



#
varlist = []
for v in dat.variables:
  # variable needs to have these two dimensions
  check_lists = ['time','site']
  if all(t in dat.variables[v].dimensions for t in check_lists):
    varlist.append(v)

print(varlist)

vars_reshaped = {}
# Reshape and append physical variables
for var in varlist:
  # this is the name of the variable.
  if np.size(dat.variables[var].shape)==3:  
      # 3-dimensional variable (time, site, lev)
      # OR (time, site, lay)
      var_dat = dat.variables[var][:,:,:]

      nlast = var_dat.shape[2]
      var_dat = var_dat.reshape(nsite_new,nlast)
      
  else: # 2-dimensional variable (time, site)
      # Extract variable and change it to shape (site)_new
      var_dat = dat.variables[var][:,:]
      var_dat = var_dat.reshape(nsite_new)
  vars_reshaped[var] = var_dat
  

# missing from this list: surface emissivity, O2 and N2
surf_emis   = dat.variables['surface_emissivity'][:]
surf_emis   = surf_emis.reshape(1, nsite).repeat(ntime,axis=0)
surf_emis   = surf_emis.reshape(nsite_new)
o2          = dat.variables['oxygen_GM'][:]
o2          = o2.reshape(ntime, 1).repeat(nsite,axis=1)
o2          = o2.reshape(nsite_new)
n2          = dat.variables['nitrogen_GM'][:]
n2          = n2.reshape(ntime, 1).repeat(nsite,axis=1)
n2          = n2.reshape(nsite_new)
vars_reshaped['surface_emissivity'] = surf_emis; varlist.append('surface_emissivity')
vars_reshaped['oxygen_GM'] = o2; varlist.append('oxygen_GM')
vars_reshaped['nitrogen_GM'] = n2; varlist.append('nitrogen_GM')

# write to new file
for var in varlist:
    var_dat = vars_reshaped[var]

    if np.size(var_dat.shape)==2:  
        # (nsite_new,lev)
        nlast = var_dat.shape[1]
        if (nlast==nlay):
            newvar = dat_new.createVariable(var,"f4",("site","layer"))
        else:
            newvar = dat_new.createVariable(var,"f4",("site","level"))    
            
        newvar[:] = var_dat[inds_keep,:]
      
    else: # (nsite_new)
        newvar = dat_new.createVariable(var,"f4",("site"))
        newvar[:] = var_dat[inds_keep]



# copy attributes

#
for varname in dat.variables:
    varin = dat.variables[varname]
    outVar = dat_new.variables[varname]
    print(varname)    
    # Copy variable attributes
    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})


# Sanity check
for v in dat_new.variables:
    print(v); 
    datt = dat_new.variables[v][:]
    print("MIN ",np.min(datt),"   MAX ",np.max(datt))  
    

dat.close()
dat_new.close()
