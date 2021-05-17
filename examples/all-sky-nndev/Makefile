# Location of RTE+RRTMGP libraries, module files.
RRTMGP_ROOT=../../
RRTMGP_BUILD = $(RRTMGP_ROOT)/build
# Sets macros FC, FCFLAGS consistent with RTE+RRTMGP
-include $(RRTMGP_BUILD)/Makefile.conf

# Location of netcdf C and Fortran libraries. Could specify with environment variables if file doesn't exist
# specified in $(RRTMGP_BUILD)/Makefile.conf
# -include $(RRTMGP_ROOT)/examples/rfmip-clear-sky/Makefile.libs
#
# RRTMGP library, module files
#
LDFLAGS   += -L$(RRTMGP_BUILD)
LIBS      += -lrrtmgp -lrte -lneural
FCINCLUDE += -I$(RRTMGP_BUILD)

#
# netcdf library, module files
# Environment variables NCHOME and NFHOME point to root of C and Fortran interfaces respectively -
#
FCINCLUDE += -I$(NFHOME)/include
LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib
LIBS      += -lnetcdff -lnetcdf


# 
# -------- BLAS lbirary: required for RRTMGP-NN --------
# 
# Choose BLAS library: e.g. "blis", "openblas", "mkl"

# Here we use BLIS (https://github.com/flame/blis) which is one of the fastest 
# BLAS libraries available (on Intel platforms MKL is probably faster). 
# the flags BLAS_DIR and BLASLIB can be specified in Makefile.conf or the environment
# Depending on your BLAS library you might have to make changes below

ifeq ($(BLASLIB),blis)
	FCINCLUDE   += -I$(BLAS_DIR)/include/blis
	LIBS      	+= $(BLAS_DIR)/lib/libblis.a -lm -lpthread 
else ifeq ($(BLASLIB),blis-amd)
	FCINCLUDE   += -I$(BLAS_DIR)/include/blis
	LIBS      	+= $(BLAS_DIR)/lib/libblis-mt.a -lm -lpthread 
else ifeq ($(BLASLIB),openblas)
	LIBS 		+= -lopenblas 
else ifeq ($(BLASLIB),mkl)
	FCINCLUDE   += -I${MKLROOT}/include
endif

#
# Setting variable USE_TIMING uses routines from the General Purpose Timing Library
#  https://jmrosinski.github.io/GPTL/
#

# Compiler specific - LDFLAGS may need to be changed
ifeq ($(GPTL_TIMING),1)
	#
	# Timing library
	#
	FCINCLUDE += -I$(TIME_DIR)/include
	# Compiler specific
	FCFLAGS += -DUSE_TIMING
	LDFLAGS   += -L$(TIME_DIR)/lib -Wl,-rpath=$(TIME_DIR)/lib
	LIBS      += -lgptl
else ifeq ($(GPTL_TIMING),2)
	#
	# Timing library
	#
	FCINCLUDE += -I$(TIME_DIR)/include
	# Compiler specific
	FCFLAGS += -DUSE_TIMING -DUSE_PAPI
	LDFLAGS   += -L$(TIME_DIR)/lib -Wl,-rpath=$(TIME_DIR)/lib
	LIBS      += -lgptl  -lpapi
endif

# -------- Use OpenMP? -fopenmp may be needed for BLAS and/or GPTL -------
ifeq ($(USE_OPENMP),1)
	# OMPFLAG = -fopenmp
	FCFLAGS += -DUSE_OPENMP
endif
OMPFLAG = -fopenmp

VPATH = ../:$(RRTMGP_ROOT)/extensions/cloud_optics

# Compilation rules
%.o: %.F90
	$(FC) $(FCFLAGS) $(FCINCLUDE) -c $< $(OMPFLAG)
%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(OMPFLAG)

#
# Extra sources -- extensions to RRTMGP classes, shared infrastructure, local sources
#
ADDITIONS  = mo_load_coefficients.o mo_simple_netcdf.o mo_cloud_optics.o mo_load_cloud_coefficients.o
ADDITIONS += mo_garand_atmos_io.o

#
# Targets
#
all: rrtmgp_allsky

rrtmgp_allsky: $(ADDITIONS) rrtmgp_allsky.o

rrtmgp_allsky.o: $(ADDITIONS) rrtmgp_allsky.F90

mo_cloud_optics.o: mo_cloud_optics.F90
mo_load_coefficients.o: mo_simple_netcdf.o mo_load_coefficients.F90
mo_garand_atmos_io.o:   mo_simple_netcdf.o mo_garand_atmos_io.F90
mo_load_cloud_coefficients.o: mo_simple_netcdf.o mo_cloud_optics.o mo_load_cloud_coefficients.F90

tests:
	cp garand-atmos-1.nc rrtmgp-allsky.nc
	$(RUN_CMD) ./rrtmgp_allsky rrtmgp-allsky.nc ${RRTMGP_ROOT}/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc ${RRTMGP_ROOT}/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc 128
	$(RUN_CMD) ./rrtmgp_allsky rrtmgp-allsky.nc ${RRTMGP_ROOT}/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ${RRTMGP_ROOT}/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc 128

check:
	python ./compare-to-reference.py

clean:
	-rm rrtmgp_allsky *.o *.optrpt ../*.optrpt *.mod
