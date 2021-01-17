#
# Here set variables RRTMGP_BUILD, NCHOME, NFHOME, TIME_DIR (for GPTL)
# and link against BLAS library, or have those variables set in the environment
#
RRTMGP_ROOT=../../
$(info VAR="$(RRTMGP_ROOT)")
RRTMGP_BUILD=$(RRTMGP_ROOT)/build
-include Makefile.libs
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


# 
# -------- BLAS lbirary: required for RRTMGP-NN --------
# 
# Choose BLAS library: e.g. "blis", "openblas", "mkl"
# 

# Here we use BLIS (https://github.com/flame/blis) which is one of the fastest 
# BLAS libraries available (on Intel platforms MKL is probably faster). 
# Depending on your BLAS library you might have to make changes below

export BLASLIB  = blis-amd
export BLAS_DIR = $(HOME)/blis-amd

# export BLASLIB = openblas
# export BLAS_DIR =  $(HOME)/blis

# ifeq ($(FC),ifort)
# 	BLASLIB=mkl
# endif

ifeq ($(BLASLIB),blis)
	FCINCLUDE   += -I$(BLAS_DIR)/include/blis
	LIBS      	+= $(BLAS_DIR)/lib/libblis.a -lm -lpthread 
else ifeq ($(BLASLIB),blis-amd)
	FCINCLUDE   += -I$(BLAS_DIR)/include/blis
	LIBS      	+= $(BLAS_DIR)/lib/libblis-mt.a -lm -lpthread 
else ifeq ($(BLASLIB),openblas)
	LIBS 		+= -lopenblas 
else ifeq ($(BLASLIB),mkl)
	# Nothing needed here if -I${MKLROOT}/include was added to FCFLAGS in build/Makefile.conf
endif

# -------- Use OpenMP? May be needed for BLAS -------
ifeq ($(BLASLIB),blis-amd)
   USE_OPENMP=1
endif
#USE_OPENMP=1

ifdef USE_OPENMP
	OMPFLAG = -fopenmp
	FCFLAGS += -DUSE_OPENMP
endif

ifdef USE_OPENACC
  FCFLAGS += -DUSE_OPENACC
endif


# Compiler specific - e.g. turn off for pgfortran, set to -cpp for gfortran
#FCFLAGS += -fpp

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
	LIBS      += -lgptl  -rdynamic  -lpapi
endif


ifeq ($(DEV_MODE),yes)
	# Modify RRTMGP interface so it outputs the neural network inputs and col_dry
	FCFLAGS += -DDEV_MODE 
endif
VPATH = ../

# Compilation rules
%.o: %.F90
	$(FC) $(FCFLAGS) $(FCINCLUDE) -c $< $(OMPFLAG)

%: %.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(OMPFLAG)

#
# Ancillary codes
#
ADDITIONS = mo_simple_netcdf.o mo_rfmip_io.o mo_load_coefficients.o

all: rrtmgp_rfmip_lw rrtmgp_rfmip_sw

rrtmgp_rfmip_lw:   rrtmgp_rfmip_lw.o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

rrtmgp_rfmip_lw.o: rrtmgp_rfmip_lw.F90 $(ADDITIONS)

rrtmgp_rfmip_sw:   rrtmgp_rfmip_sw.o   $(ADDITIONS) $(RRTMGP_BUILD)/librte.a $(RRTMGP_BUILD)/librrtmgp.a $(RRTMGP_BUILD)/libneural.a

rrtmgp_rfmip_sw.o: rrtmgp_rfmip_sw.F90 $(ADDITIONS)

mo_rfmip_io.o:          mo_rfmip_io.F90          mo_simple_netcdf.o

mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o

clean:
	-rm rrtmgp_rfmip_sw rrtmgp_rfmip_lw *.o *.mod *.optrpt
