#!/bin/make
RTE_DIR = ../rte
RRTMGP_DIR = ../rrtmgp
#
# Choose kernels depending on environment variable
#
ifeq ($(RTE_KERNELS), openacc)
  RTE_KERNEL_DIR    = ../rte/kernels-openacc
  RRTMGP_KERNEL_DIR = ../rrtmgp/kernels-openacc
endif
RTE_KERNEL_DIR += ../rte/kernels
RRTMGP_KERNEL_DIR += ../rrtmgp/kernels
NEURAL_DIR = ../neural
VPATH = $(RTE_DIR):$(RTE_KERNEL_DIR):$(RRTMGP_DIR):$(RRTMGP_KERNEL_DIR):$(NEURAL_DIR)
#
# Compiler variables FC, FCFLAGS can be set in the environment or in Makefile.conf
#
include Makefile.conf

ifeq ($(USE_TIMING),yes)
	#
	# Timing library
	FCFLAGS += -I$(TIME_DIR)/include -DUSE_TIMING -fopenmp
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
