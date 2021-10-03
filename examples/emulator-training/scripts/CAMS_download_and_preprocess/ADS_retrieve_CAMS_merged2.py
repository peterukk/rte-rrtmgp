
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve global CAMS data for radiation computations
Regridding to a coarse grid by CDS is not working currently, so data is
retrieved at the full 0.75 resolution, and then regridded
@author: Peter Ukkonen
"""

from netCDF4 import Dataset
import numpy as np
import cdsapi
import os

# timestr = ['03:00', '09:00',  '15:00',   '21:00']
# stepstr = ['3','9','15','21']
timestr = [ '09:00',   '21:00']
stepstr = ['9','21']

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_dict_eac4_sfc(year, month):
    mydict = {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
         'variable': [
            'surface_pressure',
        ],
        'time': timestr,
        
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_eac4_ml(year,month):
    mydict =     {
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
        'time': timestr,
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'variable': [
            'nitrogen_dioxide', 'ozone', 
            'carbon_monoxide', 'specific_humidity','temperature', 
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
        ],
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_egg4_sfc(year,month):
    mydict =     {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
          'variable': [
            '2m_temperature', 'forecast_albedo', 'toa_incident_solar_radiation',
        ],
        'step': stepstr,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

def get_dict_egg4_ml(year,month):
    mydict =     {
        # 'grid': [
        #     '15.0/30.0', # lat-lon 
        # ],
        'model_level': [
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', 
            '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', 
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
        ],
         'variable': [
            'methane','carbon_dioxide',
        ],
        'step': stepstr,
        'date': '%s-%s-01'%(year,month),
    }
    return mydict

# -------------------------------------------------------------------
# ---------------------- 1. DOWNLOAD MAIN CAMS DATA  -----------------
# -------------------------------------------------------------------

# dl_dir = '/media/peter/samlinux/data/CAMS/'
    
# download temp files to 
# examples/emulator-training/scripts/CAMS_download_and_preprocess/tmp
this_dir = os.getcwd() + "/"
dl_dir   = this_dir + "tmp/"

#os.chdir(this_dir)

# Specify which year to download, everything else is fixed
year = "2008"

c = cdsapi.Client()

for month in ["02", "05", "08", "11"]:
    print(month)
    
    dict_eac4 = get_dict_eac4_ml(year,month)
    c.retrieve(
        'cams-global-reanalysis-eac4', dict_eac4,
        dl_dir+'CAMS_eac4_ml_%s%s01.grb'%(year,month))
    
    dict_eac4 = get_dict_eac4_sfc(year,month)
    c.retrieve(
        'cams-global-reanalysis-eac4', dict_eac4,
        dl_dir+'CAMS_eac4_sfc_%s%s01.grb'%(year,month))
    
    # EGG4
    dict_egg4 = get_dict_egg4_ml(year,month)
    c.retrieve(
        'cams-global-ghg-reanalysis-egg4', dict_egg4,
        dl_dir+'CAMS_egg4_ml_%s%s01.grb'%(year,month))
    
    dict_egg4 = get_dict_egg4_sfc(year,month)
    c.retrieve(
        'cams-global-ghg-reanalysis-egg4', dict_egg4,
        dl_dir+'CAMS_egg4_sfc_%s%s01.grb'%(year,month))

# PREPROCESS INTO ONE NetCDF FILE USING CDO COMMANDS
# these are in a bash script, output file is tmp/CAMS_YYYY.nc
# os.system("./preproc_bash_cdo_nco {}".format(year)) 
os.system("./preproc_icon320km {}".format(year)) 


# -------------------------------------------------------------------
# ---------------------- 2. DOWNLOAD CAMS N2O PROFILES --------------
# -------------------------------------------------------------------
# Unfortunately the dataset and format is different so we need quite a lot of
# processing
fname = dl_dir+'CAMS_n2o_%s.tar.gz'%(year)

c.retrieve(
    'cams-global-greenhouse-gas-inversion',
    {
        'variable': 'nitrous_oxide',
        'quantity': 'concentration',
        'input_observations': 'surface',
        'time_aggregation': 'instantaneous',
        'version': 'latest',
        'year': '%s'%(year),
        'month': [
            '02', '05', '08',
            '11',
        ],
        'format': 'tgz',
    },
    fname)

# # Unpack and merge into on year-long file
os.chdir(dl_dir)
os.system("tar -xvzf {}".format(fname))
os.system("cdo mergetime cams73_latest_n2o_conc_surface_inst_{}*.nc tmp.nc".format(year))
# Remove ap,bp and remap to the coarser grid
os.system("ncks -O -x -v ap,bp tmp.nc tmp2.nc")
os.system("cdo remapbil,../icongrid_320km tmp2.nc CAMS_n2o_{}_tmp.nc".format(year))

# os.system("cdo remapbil,../newgrid tmp2.nc CAMS_n2o_{}_tmp.nc".format(year))
# Now add the vertical reference
os.system("ncks -A -v level,hyam,hybm,hyai,hybi ../REF_vert.nc CAMS_n2o_{}_tmp.nc".format(year))
os.system("ncrename -v Psurf,surface_air_pressure CAMS_n2o_{}_tmp.nc".format(year))
os.system("rm tmp*")
os.system("rm cams73*")
os.system("rm *.tar.gz")

# The vertical reference is inconsistent with the 3D variable
# We need to flip the profiles so they are from top to bottom of atmosphere
fname_tmp = "CAMS_n2o_{}_tmp.nc".format(year)
dat = Dataset(dl_dir+fname_tmp,'a')
sp_v = dat.variables['surface_air_pressure']
sp_v.standard_name = "surface_air_pressure"
n2o_dat = dat.variables['N2O'][:].data
n2o_dat = np.flip(n2o_dat,axis=1)
dat.variables['N2O'][:] = n2o_dat
dat.close()

# Almost done - now remap to the higher resolution vertical grid 
# used by the main CAMS data
fname_tmp2 = "CAMS_n2o_{}_tmp2.nc".format(year)
fname_n2o = "CAMS_{}_n2o.nc".format(year)
os.system("cdo remapeta,../newvct {} {}".format(fname_tmp,fname_tmp2))

# Extract the time slices corresponding to the main CAMS data
codestr = "ncks -d time,0,6,2 -d time,224,230,2 -d time,472,478,2 " \
"-dtime,720,726,2  {} {}".format(fname_tmp2,fname_n2o) 
os.system(codestr) 
os.system("rm *tmp*")

# # FINALLY, concatenate N2O and main data files, write to final destination
fname = "CAMS_{}.nc".format(year)

os.system("ncks -A {} {}".format(fname_n2o,fname))
os.system("ncatted -h -a history,global,d,, {}".format(fname))
os.system("ncatted -h -a history_of_appended_files,global,d,, {}".format(fname))

fname_final = "/media/peter/samsung/data/CAMS/CAMS_{}_2.nc".format(year)
os.system("cp {} {}".format(fname,fname_final))
