**Generate training data for ML emulators** 

Code for generating training data to replace RRTMGP and/or RTE - for research purposes

This example shortwave program loads atmospheric profiles from a netCDF file, computes gas optical properties with RRTMGP, optionally computes cloud optical properties, and then computes fluxes with RTE. The input and output from both RTE and RRTMGP can be saved to file (IO-mode). Alternatively existing neural network models can be used to replace RRTMGP (non-IO-mode).

The input data files need to have similar dimensions  ("experiment", "site"...) and variable names as the RFMIP data used in examples/rfmip-clear-sky. An RFMIP-style input file using CAMS data has been constructed and can be used for cloudy radiation computations: set include_clouds = .true. and call with `data_input/CAMS_2011_RFMIPstyle.nc`.

------

The use and compilation of RTE+RRTMGP-NN is similar to the original code, but a BLAS library is required. If you're not using ifort+MKL then [BLIS](https://github.com/flame/blis) is recommended

1. Build the RTE+RRTMGP libraries in `../../build/`. This will require setting
environment variables `FC` for the Fortran compiler and `FCFLAGS`, or creating
`../../build/Makefile.conf` with that information (see Makefile.conf.X for examples). Other optional variables include:
- (Optional) Set `GPTL=1` to use the GPTL timing library, or `GPTL=2` to use GPTL with PAPI performance counters in order to measure computational intensity. You also need to provide location `TIME_DIR`. If GPTL was built with OpenMP then you will need to add -fopenmp to compilation flags which can be done with `USE_OPENMP=1`
- (Optional) Set `USE_OPENACC=1` if you want to use OpenACC+CUDA for GPU acceleration (see Makefile.conf.nvfortran for example compilation flags)
- (Optional) Single precision is enabled by default, to use double prec. set `DOUBLE_PRECISION=1`
2. Build the executables in this directory, which will first require setting the folowing variables in the environment or via file Makefile.libs:
- (Required) The locations of the netCDF C and Fortran libraries and module files `NCHOME` and `NFHOME`
- (Required) Specify the BLAS Library (e.g. BLIS) e.g.`BLASLIB=blis` and its location `BLAS_DIR`. If you are using Intel MKL then set `BLASLIB=mkl` and ensure `MKLROOT` is specified instead. 
3. After compiling the example program, it can be used in different ways depending on arguments. Compute fluxes using reference code:

` ./nn_allsky_sw 18 data_input/inputs_RFMIP.nc`

Additionally save input output data for developing machinel learning emulators:

` ./nn_allsky_sw 18 data_input/inputs_RFMIP.nc data_ml_input_output/RFMIP_rte_rrtmgp_input_output.nc`

OR use existing ML emulator e.g. neural network to accelerate RRTMGP kernel:

` ./nn_allsky_sw 18 data_input/inputs_RFMIP.nc rrtmgp ../../neural/data/BEST_tau-sw-abs ../../neural/data/BEST_tau-sw-ray`

-----

# original instructions for rfmip-rrtmgp
This directory contains programs and support infrastructure for running
the [RTE+RRTMGP](https://github.com/RobertPincus/rte-rrtmgp) radiation parameterization for the
[RFMIP](https://www.earthsystemcog.org/projects/rfmip/) cases.

1. Build the RTE+RRTMGP libraries in `../../build/`. This will require setting
environmental variables `FC` for the Fortran compiler and `FCFLAGS`, or creating
`../../build/Makefile.conf` with that information.
2. Build the executables in this directory, which will require providing the
locations of the netCDF C and Fortran libraries and module files as environmental
variables (NCHOME and NFHOME) or via file `Makefile.libs`
3. Use Python script `stage_files.py` to download relevant files from the
[RFMIP web site](https://www.earthsystemcog.org/projects/rfmip/resources/).This script invokes another Python script to create empty output files.
4. Use Python script `run-rfmip-examples.py` to run the examples. The script takes
some optional arguments, see `run-rfmip-examples.py -h`
5. Python script `compare-to-reference.py` will compare the results to reference
answers produced on a Mac with Intel 19 Fortran compiler. Differences are normally
within 10<sup>-6</sup> W/m<sup>2</sup>.

The Python scripts require modules `netCDF4`, `numpy`, `xarray`, and `dask`.
Install with `pip` requires `pip install dask[array]` for the latter.
