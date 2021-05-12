#!/usr/bin/env make

RTE_DIR = ../rte
RRTMGP_DIR = ../rrtmgp

#
# Compiler variables FC, FCFLAGS can be set in the environment or in Makefile.conf
#
-include Makefile.conf
#
# Choose kernels depending on environment variable
#
ifeq ($(RTE_KERNELS), openacc)
  RTE_KERNEL_DIR    = ../rte/kernels-openacc
  RRTMGP_KERNEL_DIR = ../rrtmgp/kernels-openacc
endif
RTE_KERNEL_DIR += ../rte/kernels
RRTMGP_KERNEL_DIR += ../rrtmgp/kernels
VPATH = $(RTE_DIR):$(RTE_KERNEL_DIR):$(RRTMGP_DIR):$(RRTMGP_KERNEL_DIR)
#
# Compiler variables FC, FCFLAGS must be set in the environment

# Use double precision instead of single precision?
ifdef DOUBLE_PRECISION
	FCFLAGS += -DUSE_DOUBLE_PRECISION
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

all: librte.a librrtmgp.a

COMPILE = $(FC) $(FCFLAGS) $(FCINCLUDE) -c
%.o: %.F90
	$(COMPILE) $<

include $(RTE_DIR)/Make.depends
include $(RRTMGP_DIR)/Make.depends

librte.a: $(RTE_SRC)
	ar -rvs librte.a $(RTE_SRC)

librrtmgp.a: $(RRTMGP_SRC)
	ar -rvs librrtmgp.a $(RRTMGP_SRC)

clean:
	rm -f *.optrpt *.mod *.o librrtmgp.a librte.a
