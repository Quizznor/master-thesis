# General Makefile for the OfflineUser package
#
# If the environment variable AUGEROFFLINEROOT is not set
# AND the executable 'auger-offline-config' is not in your PATH
# the definition of the following variable is required
#
# AUGEROFFLINEROOT := /cr/data01/filip/offline/install
#
# Replace the wildcard expression with .cc file list if you do
# not want to compile all .cc files in this directory
#
USER_SRCS := $(wildcard *.cc)
#
# All .xml.in files will be transformed into .xml with correct
# config and schema locations
#
USER_XMLS := $(patsubst %.xml.in,%.xml,$(wildcard *.xml.in))
#
# Give your executable a name
#
EXE := userAugerOffline
#
#############################################################

## You should not need to change anything below this line ###

# Authors: T. Paul, S. Argiro, L. Nellen, D. Veberic
# $Id: Makefile.in 33888 2020-11-03 21:51:14Z darko $
# Send bug reports to http://www.auger.unam.mx/bugzilla/

.PHONY: all depend clean

ifdef AUGEROFFLINEROOT
  AUGEROFFLINECONFIG := $(AUGEROFFLINEROOT)/bin/auger-offline-config
else
  AUGEROFFLINECONFIG := auger-offline-config
  AUGEROFFLINEROOT := $(shell $(AUGEROFFLINECONFIG) --install)
endif

OBJS := $(USER_SRCS:.cc=.o)

CPPFLAGS    := $(shell $(AUGEROFFLINECONFIG) --cppflags)
CXXFLAGS    := $(shell $(AUGEROFFLINECONFIG) --cxxflags)
LDFLAGS     := $(shell $(AUGEROFFLINECONFIG) --ldflags) -lMinuit
MAIN        := $(shell $(AUGEROFFLINECONFIG) --main)
CONFIGFILES := $(shell $(AUGEROFFLINECONFIG) --config)
XMLSCHEMALOCATION := $(shell $(AUGEROFFLINECONFIG) --schema-location)

all: $(EXE) $(USER_XMLS)

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(MAIN) $(CXXFLAGS) $(LDFLAGS)

%: %.in
	@echo -n "Generating $@ file... "
	@sed -e 's!@''CONFIGDIR@!$(CONFIGFILES)!g;s!@''SCHEMALOCATION@!$(XMLSCHEMALOCATION)!g;s!@'AUGEROFFLINEROOT'@!$(AUGEROFFLINEROOT)!g' $< >$@
	@echo "done."

#############################################################
# gcc can generate the dependency list

depend: Make-depend

Make-depend: $(USER_SRCS)
	$(CPP) $(CPPFLAGS) -MM $^ >$@

clean:
	- rm -f *.o  *.so *.ps core $(USER_XMLS) Make-depend

#############################################################
# 'make run' will run the thing

run: $(EXE) $(USER_XMLS)
	./$(EXE) -b bootstrap.xml && touch $@

-include Make-depend
