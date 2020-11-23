# RTE+RRTMGP-NN is an accelerated version of RTE-RRTMGP using neural networks for the gas optics, and a refactored radiative transfer solver 
September 2020: GPU code fully working again, fixes to gas optics code to support OpenMP from an outer loop (see rrtmgp_rfmip_lw.F90)

August 2020: Shortwave NN models added. Similar performance benefit and accuracy as longwave model. Cleaner RRTMGP code: now only one interface which takes NN models as optional input to enable NN kernel. OpenACC code not tested and probably broken again.

July 2020: The neural networks now predict **molecular absorption** from which the optical depth is retrieved by multiplying with the dry air column amount Ndry. This does not change accuracy but makes the model robust to changes in vertical resolution since Ndry is no longer an input.

June 2020: RTE+RRTMGP-NN is now fully usable for the long-wave and a paper has been submitted to JAMES. Besides accelerating the long-wave gas optics computations (RRTMGP) by a factor of 2-4 by using neural networks, the solver (RTE) has been refactored to use g-points in the first dimension to be consistent with RRTMGP. This and other optimizations (e.g. Planck sources by g-point are now computed in-place in the solver) can make the solver 80% faster. When NN are additionally switched on, computing clear-sky longwave fluxes is up to 3 times faster. These results are for intel compilers and MKL - expect smaller speed-ups on other platforms and other BLAS libraries. 

------------

**How it works**: Instead of the original lookup-table interpolation routine and "eta" parameter to handle the overlapping absorption of gases in a given band, this fork implements neural networks (NNs) to predict the optical depths and planck fractions for given atmospheric conditions and gas concentrations, which includes all minor long-wave gases supported by RRTMGP. The NNs predict optical properties (molecular absorption, scattering or emission) for all spectral points from an input vector consisting of the atmospheric conditions and gas concentrations of an atmospheric layer. The models have been trained on 6-7 million samples (longwave) spanning a wide range of conditions (pre-industrial, present-day, future...) so that they may be used for both weather and climate applications. 

**Speed**: The optical depth kernel alone is 1-6 times faster while the computation of clear-sky fluxes using NNs and refactored code is 2-3 times faster. This is when all gases are included, which in the longwave results in a NN with 18 inputs in total. Expect large speedups with a fast BLAS library such as MKL and when comparing against a full computation (minor gases are expensive in the original code, "for free" with NNs"), otherwise smaller.  The NN implementation uses BLAS where the input data is packed into a (ngas * (ncol * nlay)) matrix which is then fed to GEMM call to predict a block of data at a time.
Clear-sky timings: 
<img src="https://github.com/peterukk/rte-rrtmgp-nn/blob/master/figures/figure_timings.png" width=60% height=60%>

**Accuracy**: The errors in the downwelling and upwelling fluxes are very similar to the original scheme in evaluation using RFMIP and GCM data, as well as the [CKDMIP evaluation](https://confluence.ecmwf.int/display/CKDMIP/CKDMIP%3A+Correlated+K-Distribution+Model+Intercomparison+Project+Home). 

RFMIP Heating rates:
<img src="https://github.com/peterukk/rte-rrtmgp-nn/blob/master/figures/figure_heatingrates.png" width=50% height=50%>

**Building the libraries + clear-sky example** 
The code should work very similarly to the end-user as the original, but the neural network models need to be provided at runtime: see examples/rfmip-clear-sky . Needs a fast BLAS library - if you're not using ifort+MKL then [BLIS](https://github.com/flame/blis) is recommended

1. `cd build`
2. Set required environment variables in Makefile.conf, curently using a Gfortran+BLIS platform:`FC` (the Fortran 2003 compiler),`FCFLAGS` (compiler flags). You also need specify the BLAS Library (default and recommended is BLIS) and location `BLASLIB`, `BLAS_DIR`, as well as NetCDF C and Fortran library locations  `NCHOME`,`NFHOME`. See Makefile.conf.ifort for an Intel Fortran + MKL example.
3. (Optional) Set environment variable `USE_TIMING` (to anything) to use GPTL timing library, or  `USE_FULL_TIMING` for GPTL+PAPI instrumentation. Need to also set `TIME_DIR` in Makefile.conf
4. (Optional) Set environment variable `USE_OPENACC` if you want to use OpenACC+CUDA for GPU acceleration. See Makefile.conf.nvfortran for an example.
5. `make`
6. If you have problems you might have to tinker with build/Makefile, build/Makefile.conf and examples/rfmip-clear-sky/Makefile a bit.


**to-do**

- [x] implement neural networks for shortwave
- [x] GPU kernels - should be easy and very fast with openacc_cublas **done, but note that host CUDA call overhead (such as CudaFree) was very large for small problem sizes on one tested platform (Kepler). Probably normal behaviour**
- [x] "missing gases" -how to handle these? Assume some default concentrations but what? **assumed zero by default, also present-day and pre-industrial scalar concentrations available in table, toggled in gas_optics_rrtmgp. 
- [x] offer user choice regarding speed/accuracy? (simpler, faster models which are less accurate) **tested, but as described in paper, the minor gases can be accounted for with negligible cost with NNs. The currently implemented models support CKDMIP-style gases with CFC11-eq)**
- [ ] fix cloud optics extension
- [ ] post-processing (scaling) coefficients should perhaps be integrated into neural-fortran and loaded from the same files as the model weights

# RTE+RRTMGP

This is the repository for RTE+RRTMGP, a set of codes for computing radiative fluxes in planetary atmospheres. RTE+RRTMGP is described in a [paper](https://doi.org/10.1029/2019MS001621) in [Journal of Advances in Modeling Earth Systems](http://james.agu.org).

RRTMGP uses a k-distribution to provide an optical description (absorption and possibly Rayleigh optical depth) of the gaseous atmosphere, along with the relevant source functions, on a pre-determined spectral grid given temperatures, pressures, and gas concentration. The k-distribution currently distributed with this package is applicable to the Earth's atmosphere under present-day, pre-industrial, and 4xCO2 conditions.

RTE computes fluxes given spectrally-resolved optical descriptions and source functions. The fluxes are normally summarized or reduced via a user extensible class.

Example programs and documentation are evolving - please see examples/ in the repo and Wiki on the project's Github page. Suggestions are welcome. Meanwhile for questions please contact Robert Pincus and Eli Mlawer at rrtmgp@aer.com.

## Recent changes

1. The default method for solution for longwave problems that include scattering has been changed from 2-stream methods to a re-scaled and refined no-scattering calculation following [Tang et al. 2018](https://doi.org/10.1175/JAS-D-18-0014.1).
2. In RRTMGP gas optics, the spectrally-resolved solar source function in can be adjusted by specifying the total solar irradiance (`gas_optics%set_tsi(tsi)`) and/or the facular and sunspot indicies (`gas_optics%set_solar_variability(mg_index, sb_index, tsi)`)from the [NRLSSI2 model of solar variability](http://doi.org/10.1175/BAMS-D-14-00265.1).  
3. `rte_lw()` now includes optional arguments for computing the Jacobian (derivative) of broadband flux with respect to changes in surface temperature. In calculations neglecting scattering only the Jacobian of upwelling flux is computed. When using re-scaling to account for scattering the Jacobians of both up- and downwelling flux are computed.
4. A new module, `mo_rte_config`, contains two logical variables that indicate whether arguments to routines are to be checked for correct extents and/or valid values. These variables can be changed via calls to `rte_config_checks()`. Setting the values to `.false.` removes the checks. Invalid values may cause incorrect results, crashes, or other mayhem

Relative to commit `69d36c9` to `master` on Apr 20, 2020, the required arguments to both the longwave and shortwave versions of `ty_gas_optics_rrtmgp%load()`have changed.


## Building the libraries.

1. `cd build`
2. Set environment variables `FC` (the Fortran 2003 compiler) and `FCFLAGS` (compiler flags). Alternately create a Makefile.conf that sets these variables. You could also link to an existing file.
3. Set environment variable `RTE_KERNELS` to `openacc` if you want the OpenACC kernels rather than the default.
4. `make`

## Examples

Two examples are provided, one for clear skies and one including clouds. See the README file and codes in each directory for further information.
