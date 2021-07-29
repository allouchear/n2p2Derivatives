################################################################################
#
# Library makefile
#
# Needs following variables
#
# FC  Fortran compiler
# FCFLAGS  Fortran compiler flags
# LN  Linker
# LNFLAGS  Linker flags
# SRCDIR  Folder, where source files can be found
#
################################################################################
.SUFFIX:
.SUFFIX: .f90 .o

LIBD3 = libdftd3.a
PROJLIBDIR = ../../lib/

all: $(LIBD3)

vpath % $(SRCDIR)

.PHONY: clean distclean


OBJS  = api.o common.o core.o pars.o sizes.o

$(LIBD3): $(OBJS)
	ar r $@ $^
	cp  $(LIBD3) $(PROJLIBDIR)
	cp  dftd3_api.mod $(PROJLIBDIR)

clean:
	rm -f *.o
	rm  $(PROJLIBDIR)/*.mod
	rm  $(PROJLIBDIR)/$(LIBD3)

distclean: clean
	rm -f *.mod $(LIBD3)
	rm  $(PROJLIBDIR)/*.mod
	rm  $(PROJLIBDIR)/$(LIBD3)


%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -o $@


# Dependencies
api.o: common.o core.o sizes.o
common.o:
core.o: common.o pars.o sizes.o
pars.o: common.o sizes.o
sizes.o:
