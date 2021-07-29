LIBDIR = ../../lib

FC = gfortran
CC = g++

all: addRmDFTD3

OBJS_TESTAPI =  dftd3c.o cdftd3.o AtomsProp.o

addRmDFTD3: $(OBJS_TESTAPI)
	$(CC) $(LNFLAGS) -o $@ $^ -L$(LIBDIR) -ldftd3 -lm -lgfortran

clean:
	rm -f *.o *.mod

distclean: clean
	rm -f addRmDFTD3


%.o: %.f90
	$(FC) $(FCFLAGS) -I$(LIBDIR) -c $< -o $@

%.o: %.cpp
	$(CC) -c $< -o $@

