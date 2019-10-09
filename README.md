# This fork uses neural networks to accelerate the gas optics computational kernels
# Status 4.10.2019 #
Currently only longwave implemented!!

**How it works**: instead of the original 3D interpolation routine and "eta" parameter to handle gas overlaps, this fork implements neural networks to predict the optical depths and planck fractions for a set of atmospheric conditions and gas concentrations. The model takes as input a single layer and so is agnostic to vertical discretization.  

**Speed**: The optical depth kernel is up to 3-4 times faster than the original on ifort+MKL when using single precision and a fairly complex neural network model which takes as input scaled temperature, pressure and all non-constant RRTMGP gases (19 inputs in total). The fastest implementation (on intel compilers) uses BLAS/MKL to predict multiple levels (and optionally columns) at a time using matrix-matrix products (SGEMM), reducing the whole gas optics call to a few heavy matrix operations. 

**Accuracy**: The column mean RMS errors are around 0.3, 0.5 W/m2 for up- and downwelling fluxes respectively, but the downwelling long-wave fluxes have relatively large errors up to 5 W/m2 in the upper layers at some RFMIP test sites with high water vapour concentrations (more training data needed).  The mean absolute heating rate errors (for all sites) are below 0.1 K/day except for the top levels in the stratosphere. 

**to-do**
- implement for shortwave
- GPU kernels
- post-processing (scaling) coefficients should maybe be integrated into neural-fortran and loaded from the same files as the model weights
- offer user choice regarding speed/accuracy (more accurate models are computationally slower)

# RTE+RRTMGP

This is the repository for RTE+RRTMGP, a set of codes for computing radiative fluxes in planetary atmospheres. RTE+RRTMGP is described in a [manuscript submitted Jan 17, 2019](https://owncloud.gwdg.de/index.php/s/JQo9AeRu6uIwVyR) to [Journal of Advances in Modeling Earth Systems](http://james.agu.org). 

RRTMGP uses a k-distribution to provide an optical description (absorption and possibly Rayleigh optical depth) of the gaseous atmosphere, along with the relevant source functions, on a pre-determined spectral grid given temperatures, pressures, and gas concentration. The k-distribution currently distributed with this package is applicable to the Earth's atmosphere under present-day, pre-industrial, and 4xCO2 conditions.

RTE computes fluxes given spectrally-resolved optical descriptions and source functions. The fluxes are normally summarized or reduced via a user extensible class.

Example programs and documenation are evolving - please see examples/ in the repo and Wiki on the project's Github page. Suggestions are welcome. Meanwhile for questions please contact Robert Pincus and Eli Mlawer at rrtmgp@aer.com.

## Building the libraries.

1. `cd build`
2. Create a file `Makefile.conf` defining make variables `FC` (the Fortran 2003 compiler) and `FCFLAGS` (compiler flags). Alternately  link to an existing file or set these as environment variables.
3. `make`
