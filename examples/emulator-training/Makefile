#
# Here set variables RRTMGP_BUILD, NCHOME, NFHOME, TIME_DIR (for GPTL)
# and link against BLAS library, or have those variables set in the environment
#
RRTMGP_ROOT=../../
$(info VAR="$(RRTMGP_ROOT)")
RRTMGP_BUILD=$(RRTMGP_ROOT)/build
-include $(RRTMGP_BUILD)/Makefile.conf
#
# RRTMGP library, module files
#
LDFLAGS   += -L$(RRTMGP_BUILD)
LIBS      += -lrrtmgp -lrte -lneural
FCINCLUDE += -I$(RRTMGP_BUILD)


#
# netcdf library, module files
# C and Fortran interfaces respectively
#
FCINCLUDE += -I$(NFHOME)/include
LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib
LIBS      += -lnetcdff -lnetcdf
# NetCDF-4 format for easy_netcdf
FCFLAGS += -DNC_NETCDF4


ifeq ($(DOUBLE_PRECISION),1)
	FCFLAGS += -DDOUBLE_PRECISION
endif

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
#	FCINCLUDE   += -I$(BLAS_DIR)/include/blis
	FCINCLUDE   += -I$(BLAS_DIR)/include
	LIBS      	+= $(BLAS_DIR)/lib/libblis-mt.a -lm -lpthread 
else ifeq ($(BLASLIB),openblas)
	LIBS 		+= -lopenblas 
else ifeq ($(BLASLIB),mkl)
	FCINCLUDE   += -I${MKLROOT}/include
endif


# -------- Use OpenMP? May be needed for BLAS -------
# ifeq ($(BLASLIB),blis-amd)
#    USE_OPENMP=1
# endif
#USE_OPENMP=1

ifeq ($(USE_OPENMP),1)
	# OMPFLAG = -fopenmp
	FCFLAGS += -DUSE_OPENMP
endif
OMPFLAG = -fopenmp

ifeq ($(USE_OPENACC),1)
  FCFLAGS += -DUSE_OPENACC
endif

# -------- GPTL Timing -------
# Use General Purpose Timing Library to profile code?
ifeq ($(GPTL_TIMING),1)
	#
	# Timing library
	#
	FCINCLUDE += -I$(TIME_DIR)/include
	# Compiler specific
	FCFLAGS += -DUSE_TIMING
	LDFLAGS   += -L$(TIME_DIR)/lib -Wl,-rpath=$(TIME_DIR)/lib
	LIBS      += -lgptl
# Use GPTL together with PAPI hardware instrumentation to measure computational intensity?
# Requires that your GPTL was compiled to PAPI support
else ifeq ($(GPTL_TIMING),2)
	#
	# Timing library
	#
	FCINCLUDE += -I$(TIME_DIR)/include
	# Compiler specific
	FCFLAGS += -DUSE_TIMING -DUSE_PAPI
	LDFLAGS   += -L$(TIME_DIR)/lib -Wl,-rpath=$(TIME_DIR)/lib
	# LIBS      += -lgptl  -rdynamic  -lpapi
	LIBS      += -lgptl -lpapi
ifeq ($(FC),gfortran)
	LIBS      += -rdynamic  
endif
endif

VPATH = ../:$(RRTMGP_ROOT)/extensions/cloud_optics


# Compilation rules
%.o: %.F90
	$(FC) $(FCFLAGS) $(FCINCLUDE) -c $< $(OMPFLAG)

%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(OMPFLAG)

#
# Ancillary codes
#
ADDITIONS = mo_simple_netcdf.o easy_netcdf.o mo_io_rfmipstyle_generic.o mo_load_coefficients.o mo_cloud_optics.o mo_load_cloud_coefficients.o

all: allsky_sw_gendata allsky_sw_testmodels

allsky_sw_gendata:   allsky_sw_gendata.o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

allsky_sw_gendata.o: allsky_sw_gendata.F90 $(ADDITIONS)

allsky_sw_testmodels:   allsky_sw_testmodels.o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

allsky_sw_testmodels.o: allsky_sw_testmodels.F90 $(ADDITIONS)

easy_netcdf.o: easy_netcdf.F90

mo_io_rfmipstyle_generic.o: mo_io_rfmipstyle_generic.F90 mo_simple_netcdf.o easy_netcdf.o

mo_cloud_optics.o: mo_cloud_optics.F90

mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o

mo_load_cloud_coefficients.o: mo_simple_netcdf.o mo_cloud_optics.o mo_load_cloud_coefficients.F90


clean:
	-rm allsky_sw_gendata allsky_sw_testmodels *.o *.mod *.optrpt
