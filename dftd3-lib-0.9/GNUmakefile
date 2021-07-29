.SUFFIX:
.SUFFIX: .f90 .o

.PHONY: dftd3 lib testapi

all: lib 

include make.arch

lib:
	$(MAKE) -C lib FC="$(FC)" FCFLAGS="$(FCFLAGS)" LN="$(LN)" \
            LNFLAGS="$(LNFLAGS)" SRCDIR="."

.PHONY: clean distclean
clean:
	$(MAKE) -C lib clean

distclean:
	$(MAKE) -C lib distclean
