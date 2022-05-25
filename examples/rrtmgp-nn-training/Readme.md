**Train neural network emulators for RRTMGP** 

Code for generating training data and train NN versions of RRTMGP k-distributions

Goal is cleaner and more generic code than last time, but contributions for further improvements are welcome.

After running make, data can be generated like this

`./rrtmgp_sw_gendata_rfmipstyle [block_size] [input file] [k-distribution file] [input-output file]`

For instance, to generate data from the 1800 RFMIP profiles, using the reduced k-distributions: 

`./rrtmgp_sw_gendata_rfmipstyle 8 ../rfmip-clear-sky/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ../../rrtmgp/data/rrtmgp-data-sw-g112-210809.nc $ML_DATA_FOLDER/ml_training_sw_g112_RMFIP.nc`

`./rrtmgp_lw_gendata_rfmipstyle 8 ../rfmip-clear-sky/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ../../rrtmgp/data/rrtmgp-data-lw-g128-210809.nc $ML_DATA_FOLDER/ml_training_lw_g128_RMFIP.nc`

In the paper, the RFMIP profiles were not used for training, instead training data was generated from the input files in inputs_to_RRTMGP.zip

The output netCDF files (last argument) are used by the training program `ml_train.py` and can be rather big, as they contain RRTMGP inputs (4D array, with features innermost) and outputs (4D, with g-points innermost), where the 3 outer dimensions contain different perturbation experiments, columns and levels and are collapsed before training:

```
dimensions:
	expt = 18 ;
	site = 100 ;
	layer = 60 ;
	level = 61 ;
	feature = 7 ;
	gpt = 112 ;
	bnd = 14 ;
variables:
	float rsu(expt, site, level) ;
		rsu:long_name = "upwelling shortwave flux" ;
	float rsd(expt, site, level) ;
		rsd:long_name = "downwelling shortwave flux" ;
	float rsd_dir(expt, site, level) ;
		rsd_dir:long_name = "direct downwelling shortwave flux" ;
	float pres_level(expt, site, level) ;
		pres_level:long_name = "pressure at half-level" ;
	float rrtmgp_sw_input(expt, site, layer, feature) ;
		rrtmgp_sw_input:long_name = "inputs for RRTMGP shortwave gas optics" ;
		rrtmgp_sw_input:comment = "tlay play h2o o3 co2 n2o ch4" ;
	float tau_sw_gas(expt, site, layer, gpt) ;
		tau_sw_gas:long_name = "gas optical depth" ;
	float ssa_sw_gas(expt, site, layer, gpt) ;
		ssa_sw_gas:long_name = "gas single scattering albedo" ;
	float col_dry(expt, site, layer) ;
		col_dry:long_name = "layer number of dry air molecules" ; 
```

The input data (second argument) can come from anywhere as long as it follows the RFMIP syntax when it comes to names of dimensions ("sites" represent columns, "experiment" can represent different climate or gas perturbation experiments - can of course have a length of 1), names of dimensions, as well as order of dimensions. However, gases can be provided as either scalar/1D with no height dependency (expt), 2D field with no height dependency (expt, site), or 3D field (expt, site, layer).

A separate traning program can then collapse inputs and outputs into 2D fields *x_raw* (nsamples, nfeatures) and *y_raw* (nsamples,ngpt), apply some pre-processing *x_raw* -> *x*, *y_raw* -> *y*, and train on (*x*,*y*). To generalize to arbitrary vertical grids it is recommended to normalize optical depths *tau* by the layer number of molecules *N* ("col_dry"). Predicting absorption and Rayleigh cross-sections in the shortwave (instead of total optical depth and single-scattering albedo) gets one even closer to the "physics" / inner workings of the original code:

y<sub>abs</sub> = (tau - tau<sub>ray</sub>) / N = (tau - tau * ssa) / N

y<sub>ray</sub> = (tau * ssa) / N 

Similarly, predicting Planck fraction as the emission variable is recommended, from which upward and downward Planck functions can then be computed. These physical scalings have shown to be effective; in addition, it may be worthwhile to use general ML preprocessing methods ("standardization", "normalization"), to obtain inputs and outputs 1) in a similar in a similar range to other inputs/outputs, 2) not too big or large (e.g. 0-1 instead of 0-1 * 10e-5) and 3) have a more normal distribution than the raw variable (e.g. log(p), instead of p) - see `ml_train.py`