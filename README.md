# This fork uses neural networks to accelerate the gas optics computational kernels
Status 31.1.2020: RTE code rewritten to use g-points in the first dimension, and other optimizations which can lead to a 40% speedup overall (without neural networks). Coming soon!

# Currently only longwave implemented!!

**How it works**: instead of the original 3D interpolation routine and "eta" parameter to handle the overlapping absorption of "minor" gases in a given band, this fork implements neural networks to predict the optical depths and planck fractions for a set of atmospheric conditions and gas concentrations, which includes a large number of absorbing gases (17). The model takes as input a single layer and so is agnostic to vertical discretization.  

**Speed**: The optical depth kernel is up to 3-4 times faster than the original on ifort+MKL when using single precision and a 4-layer neural network model which takes as input scaled temperature, pressure and all non-constant RRTMGP gases (19 inputs in total). Optical depths and planck fractions are predicted by separate models, which output all 256 g-points. The fastest implementation uses BLAS/MKL where the data is packed into a matrix which is then fed to GEMM call to predict a block of data at a time (neural networks = vector/vector or matrix/matrix products).

**Accuracy**: The mean absolute and max vertical errors in the downwelling and up-welling fluxes are now comparable to the original scheme (<0.5 W/m2 for mean, 1-2 W/m2 for max vertical error) relative to an accurate line-by-line model. The upwelling fluxes are in some cases (surprisingly) even more accurate; however the heating rates still end up being somewhat less accurate overall due to being sensitive to flux errors in the stratosphere, where the neural network performs worse. These results are based on a pseudo-independent test set (where the temperature and humidity profiles are independent but not  the combination of gas concentrations).

**how to use** 

**to-do**
- "missing gases" -how to handle these? Assume some default concentrations but what? A range of models for various use cases (e.g. GCM, GCM-lite, NWP...)?
- related to this, offer user choice regarding speed/accuracy (more accurate models are computationally slower)
- implement for shortwave
- GPU kernels
- post-processing (scaling) coefficients should maybe be integrated into neural-fortran and loaded from the same files as the model weights


# RTE+RRTMGP

This is the repository for RTE+RRTMGP, a set of codes for computing radiative fluxes in planetary atmospheres. RTE+RRTMGP is described in a [paper](https://doi.org/10.1029/2019MS001621) in [Journal of Advances in Modeling Earth Systems](http://james.agu.org).

RRTMGP uses a k-distribution to provide an optical description (absorption and possibly Rayleigh optical depth) of the gaseous atmosphere, along with the relevant source functions, on a pre-determined spectral grid given temperatures, pressures, and gas concentration. The k-distribution currently distributed with this package is applicable to the Earth's atmosphere under present-day, pre-industrial, and 4xCO2 conditions.

RTE computes fluxes given spectrally-resolved optical descriptions and source functions. The fluxes are normally summarized or reduced via a user extensible class.

Example programs and documenation are evolving - please see examples/ in the repo and Wiki on the project's Github page. Suggestions are welcome. Meanwhile for questions please contact Robert Pincus and Eli Mlawer at rrtmgp@aer.com.

In the most recent revision, the default method for solution for longwave problems that include scattering has been changed from 2-stream methods to a re-scaled and refined no-scattering calculation following [Tang et al. 2018](https://doi.org/10.1175/JAS-D-18-0014.1).

## Building the libraries.

1. `cd build`
2. Set environment variables `FC` (the Fortran 2003 compiler) and `FCFLAGS` (compiler flags). Alternately create a Makefile.conf that sets these variables. You could also link to an existing file.
3. Set environment variable `RTE_KERNELS` to `openacc` if you want the OpenACC kernels rather than the default.
4. `make`

## Examples

Two examples are provided, one for clear skies and one including clouds. See the README file and codes in each directory for further information.
