#
# Here set variables RRTMGP_DIR, NCHOME, NFHOME, TIME_DIR (for GPTL)
#   or have those variables set in the environment
#
-include Makefile.libs
-include $(RRTMGP_DIR)/Makefile.conf
#
# RRTMGP and NEURAL-FORTRAN libraries, module files
#
LDFLAGS   += -L$(RRTMGP_DIR)
LIBS      += -lrrtmgp -lrte -lneural
FCINCLUDE += -I$(RRTMGP_DIR)

#
# netcdf library, module files
# C and Fortran interfaces respectively
#
FCINCLUDE += -I$(NFHOME)/include
LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib  -L/usr/local/lib 
LIBS      += -lnetcdff -lnetcdf 

#
# Setting macro TIMING=yes uses routines from the General Purpose Timing Library
#  https://jmrosinski.github.io/GPTL/
#
# THIS IS SET UP IN THE MASTER MAKEFILE

# Use OpenMP?
ifeq ($(OMP),yes)
	OMPFLAG = -fopenmp
endif

# 
# Choose BLAS library: e.g. "blis", "openblas", "mkl"
# 
BLASLIB=blis

ifeq ($(FC),ifort)
	BLASLIB=mkl
endif

ifeq ($(BLASLIB),blis)
	# BLIS_PREFIX 	= /usr/local
	# BLIS_INCLUDE 	= $(BLIS_PREFIX)/include/blis
	# BLIS_LIB   	= $(BLIS_PREFIX)/lib
	# FCINCLUDE 	+= -I$(BLIS_INCLUDE)
	# LDFLAGS   	+= -L$(BLIS_LIB) 
	# LIBS      	+= -lm -lpthread -lblis

	# assuming the libraries are installed in standard locations
	LIBS 		+= -lblis
else ifeq ($(BLASLIB),openblas)
	LIBS 		+= -lopenblas 
else ifeq ($(BLASLIB),mkl)
	
endif

# Compiler specific - e.g. turn off for pgfortran, set to -cpp for gfortran

ifeq ($(USE_TIMING),yes)
	#
	# Timing library
	#
	FCINCLUDE += -I$(TIME_DIR)/include
	# Compiler specific
	FCFLAGS += -DUSE_TIMING
	LDFLAGS   += -L$(TIME_DIR)/lib
	LIBS      += -lgptl
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

rrtmgp_rfmip_lw:   rrtmgp_rfmip_lw.o   $(ADDITIONS) $(RRTMGP_DIR)/librte.a $(RRTMGP_DIR)/librrtmgp.a $(RRTMGP_DIR)/libneural.a

rrtmgp_rfmip_lw.o: rrtmgp_rfmip_lw.F90 $(ADDITIONS)

rrtmgp_rfmip_sw:   rrtmgp_rfmip_sw.o   $(ADDITIONS) $(RRTMGP_DIR)/librte.a $(RRTMGP_DIR)/librrtmgp.a $(RRTMGP_DIR)/libneural.a

rrtmgp_rfmip_sw.o: rrtmgp_rfmip_sw.F90 $(ADDITIONS)

mo_rfmip_io.o:          mo_rfmip_io.F90          mo_simple_netcdf.o

mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o

clean:
	-rm rrtmgp_rfmip_sw rrtmgp_rfmip_lw *.o *.mod *.optrpt
