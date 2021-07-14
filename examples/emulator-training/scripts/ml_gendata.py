#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of RRTMGP gas optics
scheme, the RTE radiative transfer solver, or their combination RTE+RRTMGP (a 
radiative transfer scheme).

This script (run from examples/emulator-training/scripts/) generates training 
data by calling a RTE+RRTMGP program with an RFMIP-style netCDF input data file, 
and saves training data (inputs and outputs of the different components) to 
another file, which can then be used for training emulators.

The RTE+RRTMGP program, ml_allsky_sw.F90, does two-stream shortwave radiation 
computations with scattering and includes clouds by default (can be turned
off inside the program), which requires cloud fraction, specific cloud ice and 
liquid water contents (tested with CAMS data).

You also need to modify ml_allsky_sw if you want to change which gases are used

@author: Peter Ukkonen
"""



import os, subprocess, argparse
from ml_loaddata import *

#from gasopt_load_train_funcs import load_data_all,create_model, gptnorm_numba,gptnorm_numba_reverse
#from gasopt_load_train_funcs import ymeans_lw, ysigma_lw, ymeans_sw, ysigma_sw, ysigmas_sw, ysigmas_lw
#from gasopt_load_train_funcs import ymeans_sw_ray, ysigma_sw_ray, ymeans_sw_abs, ysigma_sw_abs
#from gasopt_load_train_funcs import plot_hist2d_T, plot_hist2d

import gc
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------------------------------------------------------
# ----------------- SET input and output file names  -----------------
input_file          = "CAMS_2011_RFMIPstyle.nc"            
output_prefix       = "ml_data_g224_clouds_"                 
# ----------------- SET RRTMGP gas coefficients file  -----------------
sw_gas_coeffs_file  = "rrtmgp-data-sw-g224-2018-12-04.nc"  
# ----------------- SET block size (number of columns)  -----------------
block_size          = 480                           


#  ----------------- File paths -----------------
#this_dir        = os.getcwd()
this_dir = ""
rte_rrtmgp_dir  = this_dir + "../../../"
emulator_dir    = this_dir + "../"

input_path  = emulator_dir + "data_input/" + input_file
output_file = output_prefix + os.path.splitext(input_file)[0] + ".nc"
output_path = emulator_dir + "data_training/" + output_file

sw_gas_coeffs_path = rte_rrtmgp_dir + "rrtmgp/data/" + sw_gas_coeffs_file

fortran_exe_name = emulator_dir + "./ml_allsky_sw"

#  ml_allsky_sw [block_size] [input file] [k-distribution file] [input/output file for NN development] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs ml-emulator example, saving data for ml training. No arguments, modify script instead")
#    parser.add_argument("--run_command", type=str, default="",
#                        help="Prefix ('jsrun' etc.) for running commands. Use quote marks to enclose multi-part commands.")
#    parser.add_argument("--block_size", type=int, default=8,
#                        help="Number of columns to compute at a time. Must be a factor of 1800 (ncol*nexp)")

    args = parser.parse_args()
    block_size_str   = '{0:4d}'.format(block_size)
#    if args.run_command:
#        print ("using the run command")
#        rfmip_sw_exe_name = args.run_command + " " + rfmip_sw_exe_name

    print("Running " + fortran_exe_name + block_size_str + " " + input_path + " " + sw_gas_coeffs_path + " " + output_path)
    # arguments are block size, input conditions, coefficient file, output file
    subprocess.run([fortran_exe_name, block_size_str, input_path, sw_gas_coeffs_path, output_path])

