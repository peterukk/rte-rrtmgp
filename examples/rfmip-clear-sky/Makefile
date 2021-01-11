#
# Here set variables RRTMGP_DIR, NCHOME, NFHOME, TIME_DIR (for GPTL)
#   or have those variables set in the environment
#
RRTMGP_BUILD=$(RRTMGP_ROOT)/build
-include Makefile.libs
-include $(RRTMGP_DIR)/Makefile.conf
#
# RRTMGP libraries, module files
#
LDFLAGS   += -L$(RRTMGP_DIR)
LIBS      += -lrrtmgp -lrte 
FCINCLUDE += -I$(RRTMGP_DIR)

#
# netcdf library, module files
# C and Fortran interfaces respectively
#
FCINCLUDE += -I$(NFHOME)/include
LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib
LIBS      += -lnetcdff -lnetcdf

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
	LIBS      += -lgptl  -rdynamic  -lpapi
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

rrtmgp_rfmip_lw:   rrtmgp_rfmip_lw.o   $(ADDITIONS) $(RRTMGP_DIR)/librte.a $(RRTMGP_DIR)/librrtmgp.a

rrtmgp_rfmip_lw.o: rrtmgp_rfmip_lw.F90 $(ADDITIONS)

rrtmgp_rfmip_sw:   rrtmgp_rfmip_sw.o   $(ADDITIONS) $(RRTMGP_DIR)/librte.a $(RRTMGP_DIR)/librrtmgp.a

rrtmgp_rfmip_sw.o: rrtmgp_rfmip_sw.F90 $(ADDITIONS)

mo_rfmip_io.o:          mo_rfmip_io.F90          mo_simple_netcdf.o

mo_load_coefficients.o: mo_load_coefficients.F90 mo_simple_netcdf.o

clean:
	-rm rrtmgp_rfmip_sw rrtmgp_rfmip_lw *.o *.mod *.optrpt
