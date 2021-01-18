**Building the libraries and running clear-sky example with RFMIP-RRTMGP-NN** 

The use and compilation of RTE+RRTMGP-NN is similar to the original code, but a BLAS library is required. If you're not using ifort+MKL then [BLIS](https://github.com/flame/blis) is recommended

1. Build the RTE+RRTMGP libraries in `../../build/`. This will require setting
environment variables `FC` for the Fortran compiler and `FCFLAGS`, or creating
`../../build/Makefile.conf` with that information (see Makefile.conf.X for examples). Other optional variables include:
- (Optional) Set `GPTL=1` to use the GPTL timing library, or `GPTL=2` to use GPTL with PAPI performance counters in order to measure computational intensity. You also need to provide location `TIME_DIR`. If GPTL was built with OpenMP then you will need to add -fopenmp to compilation flags which can be done with `USE_OPENMP=1`
- (Optional) Set `USE_OPENACC=1` if you want to use OpenACC+CUDA for GPU acceleration (see Makefile.conf.nvfortran for example compilation flags)
- (Optional) For even more speed on non-Intel platforms (40-200% faster solver on GNU), set `FAST_EXPONENTIAL=1` to use an approximation to the exponential function. This only lead to a max. 0.2 W/m2 deviation in net shortwave fluxes and much less in the longwave 
- (Optional) Single precision is enabled by default and recommended, to use double prec. set `DOUBLE_PRECISION=1`
2. Build the executables in this directory, which will first require setting the folowing variables in the environment or via file Makefile.libs:
- (Required) The locations of the netCDF C and Fortran libraries and module files `NCHOME` and `NFHOME`
- (Required) Specify the BLAS Library (e.g. BLIS) e.g.`BLASLIB=blis` and its location `BLAS_DIR`. If you are using Intel MKL then set `BLASLIB=mkl` and ensure `MKLROOT` is specified instead. 
3. After compiling the clear-sky examples, they can be run either via `run-rfmip-examples.py` or manually:

` ./rrtmgp_rfmip_lw 8 multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc 1 1`

` ./rrtmgp_rfmip_sw 8 multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc 1 1`

In this branch the input (multiple_input..) and output files (in subfolder output_fluxes) are already present so no Python script is needed to acquire them.
The example programs have been modified to allow neural networks to be used by setting `use_nn = .true.` in rrttmgp_rfmip_X.F90. In addition, there is an option (`compare_flux =.true.`) to compare the output fluxes to benchmark line-by-line computations, alongside reference RRTMGP results which were produced in double precision.

4. (Optional) if GPTL was enabled, inspect the timing results `cat timing.lw-8 ; cat timing.sw-8`.


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
