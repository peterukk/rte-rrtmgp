# RTE+RRTMGP-NN is an accelerated version of RTE-RRTMGP using neural networks for the gas optics computations 

Update 4.6.2020: RTE+RRTMGP-NN is now fully usable for the long-wave and a paper is being written. Besides accelerating the long-wave gas optics computations (RRTMGP) by a factor of 2-4 by using neural networks, the solver (RTE) has been rewritten to use g-points in the first dimension to be consistent with RRTMGP. This and other optimizations (e.g. Planck sources by g-point are now computed in-place in the solver) lead to almost 80% speedup overall without neural networks, while the whole clear-sky radiative transfer is more than 3 times faster when neural networks are used. These results are for an Intel platform using the latest version of MKL - expect smaller speed-ups on other platforms and other BLAS libraries). 

No neural network has been developed for the **shortwave** yet. Because of the refactoring, also the shortwave code is faster (but the fluxes differ slightly in single precision?)

The **cloud optics** extension is still broken.

**GPU** acceleration is supported (openACC+cuBLAS), but the code is currently slow due to non-computational bottlenecks (working on it). 

------------

**How it works**: instead of the original 3D interpolation routine and "eta" parameter to handle the overlapping absorption of "major" gases in a given band, this fork implements neural networks to predict the optical depths and planck fractions for given atmospheric conditions and gas concentrations, which includes all minor long-wave gases supported by RRTMGP. The neural network predicts optical properties (optical depth or Planck fraction) for all 256 g-points from one input vector (the atmospheric conditions for one atmospheric layer), therefore avoiding loops over g-point or band. The model has been trained on very diverse data so that it may be used for both weather and climate applications. 

**Speed**: The optical depth kernel is up to 4 times faster than the original on ifort+MKL when using single precision and neural network with 2 hidden layers which takes as input scaled temperature, pressure and all non-constant RRTMGP gases (19 inputs in total) and predict optical depth and planck fraction (256 outputs), using two separate models. The fastest implementation uses BLAS/MKL where the input data is packed into a (ngas * (ncol * nlay)) matrix which is then fed to GEMM call to predict a block of data at a time (replacing the matrix-vector dot product of a feed-forward neural network with a matrix-matrix call).

**Accuracy**: The errors in the downwelling and up-welling fluxes are similar to the original scheme in the tests done so far using RFMIP and GCM data. CKDMIP evaluation coming soon. 

**how to use** 
The code should work very similarly to the end-user as the original, but the neural network models need to be provided at runtime: see examples/rfmip-clear-sky . Needs a fast BLAS library - if you're not using ifort+MKL then [BLIS](https://github.com/flame/blis) is recommended

**to-do**
- "missing gases" -how to handle these? Assume some default concentrations but what? A range of models for various use cases (e.g. GCM, GCM-lite, NWP...)? **done**
- related to this, offer user choice regarding speed/accuracy? (simpler, faster models which are less accurate) **simpler models using less gases do not seem much faster, but the code now supports using less gases as input (CKDMIP-gases only with CFC11-eq) - these models need to be updated**
- implement neural networks for shortwave
- GPU kernels - should be easy and very fast with openacc_cublas **done, but the code is slow due to spurious CUDA memory and deallocations on the device. needs to be looked into**
- post-processing (scaling) coefficients should perhaps be integrated into neural-fortran and loaded from the same files as the model weights


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
