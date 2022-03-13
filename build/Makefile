#!/bin/make
RTE_DIR = ../rte
RRTMGP_DIR = ../rrtmgp

#
# Compiler variables FC, FCFLAGS can be set in the environment or in Makefile.conf
#
-include Makefile.conf
#
# Choose kernels depending on environment variable
#

ifeq ($(USE_OPENACC),1)
  RTE_KERNEL_DIR    = ../rte/kernels-openacc
  RRTMGP_KERNEL_DIR = ../rrtmgp/kernels-openacc
  FCFLAGS += -DUSE_OPENACC
endif

# Allow inner dimension (number of g-points) to be known at compile time
ifdef NGPT_SW
FCFLAGS += -DNGPT_SW=$(NGPT_SW)
endif
ifdef NGPT_LW
FCFLAGS += -DNGPT_LW=$(NGPT_LW)
endif
ifdef NLAY
FCFLAGS += -DNLAY=$(NLAY)
endif
#
# netcdf library, module files
# C and Fortran interfaces respectively
#
FCINCLUDE += -I$(NFHOME)/include
# LDFLAGS   += -L$(NFHOME)/lib -L$(NCHOME)/lib
# LIBS      += -lnetcdff -lnetcdf



RTE_KERNEL_DIR += ../rte/kernels
RRTMGP_KERNEL_DIR += ../rrtmgp/kernels
NEURAL_DIR = ../neural
EXAMPLES_DIR = ../examples
VPATH = $(RTE_DIR):$(RTE_KERNEL_DIR):$(RRTMGP_DIR):$(RRTMGP_KERNEL_DIR):$(NEURAL_DIR):$(EXAMPLES_DIR)

ifeq ($(DOUBLE_PRECISION),1)
	FCFLAGS += -DDOUBLE_PRECISION
endif

# Use GPTL timing library? Added this to main makefile to enable
# timing instrumentation within RRTMGP and RTE source code
ifeq ($(GPTL_TIMING),1)
	# Timing library GPTL
	FCFLAGS += -I$(TIME_DIR)/include -DUSE_TIMING 

# Use GPTL with PAPI for hardware performance counters? (enables measuring compute intensity)
else ifeq ($(GPTL_TIMING),2)
	FCFLAGS += -I$(TIME_DIR)/include -DUSE_TIMING -DUSE_PAPI
endif

all: librte.a librrtmgp.a libneural.a

COMPILE = $(FC) $(FCFLAGS) $(FCINCLUDE) -c
%.o: %.F90
	$(COMPILE) $<

include $(RTE_DIR)/Make.depends
include $(RRTMGP_DIR)/Make.depends
include $(NEURAL_DIR)/Make.depends

librte.a: $(RTE_SRC)
	ar -rvs librte.a $(RTE_SRC)

librrtmgp.a: $(RRTMGP_SRC)
	ar -rvs librrtmgp.a $(RRTMGP_SRC)

libneural.a: $(NEURAL_SRC)
	ar -rvs libneural.a $(NEURAL_SRC)

clean:
	rm -f *.optrpt *.mod *.o librrtmgp.a librte.a libneural.a
