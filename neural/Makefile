#LIBDIR=/usr/lib/x86_64-linux-gnu/
#LDFLAGS += -L$(LIBDIR)
#FCINCLUDE = -I$(LIBDIR)
# VPATH = $(LIBDIR)
# VPATH = ../examples
-include ../Makefile.conf

all: libneural.a

COMPILE = $(FC) $(FCFLAGS) $(FCINCLUDE) -c
%.o: %.F90
	$(COMPILE) $<
# above means: make all .o files from the corresponding .F90, recompiling any .F90 file that is newer
# than the .o file ($<) 
# should FCFLAGS come after -c?
	
# include the file listing the library of subprograms (the object files consisting of inter-dependencies) in variable NEURAL_SRC
include Make.depends

# create a static library for NEURAL by using ar (NEURAL_SRC from $NEURAL_DIR/Make.depends includes the dependencies)
libneural.a: $(NEURAL_SRC)
	ar rcs libneural.a $(NEURAL_SRC)

# The rvs flags tell ar to put named files into the library, replacing any previous versions of these files if necessary, print the names of object files as it puts them into the library, and add an object-file index to the library, which makes it link faster


