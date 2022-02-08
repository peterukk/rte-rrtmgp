# Set k-distribution files
# KDIST_LW=rrtmgp-data-lw-g256-2018-12-04.nc
# KDIST_SW=rrtmgp-data-sw-g224-2018-12-04.nc
KDIST_LW=rrtmgp-data-lw-g128-210809.nc
KDIST_SW=rrtmgp-data-sw-g112-210809.nc

BLOCKSIZE = 36
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
	#LDFLAGS     += -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl
endif


# -------- Use OpenMP? May be needed for BLAS -------

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

VPATH = ../

PROGRAM1 = rrtmgp_sw_gendata_rfmipstyle
PROGRAM2 = rrtmgp_lw_gendata_rfmipstyle
PROGRAM3 = rrtmgp_sw_eval_nn_rfmip
PROGRAM4 = rrtmgp_lw_eval_nn_rfmip

# Compilation rules
%.o: %.F90
	$(FC) $(FCFLAGS) $(FCINCLUDE) -c $< $(OMPFLAG)

%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(OMPFLAG)

#
# Ancillary codes
#
ADDITIONS = mo_simple_netcdf.o easy_netcdf.o mo_io_rfmipstyle_generic.o mo_load_coefficients.o

all: $(PROGRAM1) $(PROGRAM2) $(PROGRAM3) $(PROGRAM4)


$(PROGRAM1) $(PROGRAM2) $(PROGRAM3) $(PROGRAM4): % : %.o  $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

# $(PROGRAM1).o $(PROGRAM2).o: % : %.F90 $(ADDITIONS)

# $(PROGRAM1):   $(PROGRAM1).o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

$(PROGRAM1).o: $(PROGRAM1).F90 $(ADDITIONS)

# $(PROGRAM2):   $(PROGRAM2).o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

$(PROGRAM2).o: $(PROGRAM2).F90 $(ADDITIONS)
$(PROGRAM3).o: $(PROGRAM3).F90 $(ADDITIONS)
$(PROGRAM4).o: $(PROGRAM4).F90 $(ADDITIONS)


easy_netcdf.o: easy_netcdf.F90

mo_io_rfmipstyle_generic.o:          mo_io_rfmipstyle_generic.F90          mo_simple_netcdf.o

mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o

tests: multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc # Also the template files
	# Files need to have been generated/downloaded before
	$(RUN_CMD) ./$(PROGRAM1) $(BLOCKSIZE) multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ${RRTMGP_ROOT}/rrtmgp/data/$(KDIST_LW)
	$(RUN_CMD) ./$(PROGRAM2) $(BLOCKSIZE) multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc ${RRTMGP_ROOT}/rrtmgp/data/$(KDIST_SW)

check:
	cd ${RRTMGP_ROOT}/examples/rfmip-clear-sky && python ./compare-to-reference.py --fail=7.e-4

multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc:
	bash ./stage_files.sh

clean:
	-rm $(PROGRAM2) $(PROGRAM1) *.o *.mod *.optrpt
