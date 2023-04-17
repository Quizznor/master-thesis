# GPLv2 and 2C-BSD
# Copyright (c) Darko Veberic, 2014

# CXX := clang
#CPPFLAGS := -I. $(shell auger-offline-config --cppflags) -DDEBUG
# TODO: remove this hack
hdf5 := $(shell auger-offline-config --cppflags | awk -v FS='External' '/Geant4/{print $$1}')

CPPFLAGS := -I. -I./src $(shell auger-offline-config --cppflags) \
						$(hdf5)External/hdf5/1.10.1/include

# make CXX=/cr/data01/hahn/auger/External/hdf5/1.10.1/bin/h5c++ && ./read_adst_tests ADST_helium_19_19.5_dat100712.root
# 

#CXXFLAGS := -Wall -Wextra -Wpedantic -std=c++11 -ggdb3 -O0 -fno-inline -pipe
# CXXFLAGS := -Wall -Wextra -std=c++14 -pipe -g -O0  # -ggdb3 -O2 -fPIC

# CXXFLAGS := -ggdb3 -Wall -Wextra -Wtype-limits -std=c++14 -O0 \
# -fno-inline -fno-omit-frame-pointer -fsanitize=address 
 # asan clang

CXXFLAGS := -ggdb3 -Wall -Wextra -Wtype-limits -std=c++17 -O0 \
            -fno-inline -fno-omit-frame-pointer -fsanitize=address \
# -fsanitize=bounds-strict \
# -pipe  # asan gcc


LDFLAGS := $(shell auger-offline-config --ldflags)

SHELL := bash
SUFFIXES := .o .cc .cxx

EXES := $(basename $(wildcard *.cxx))
OBJS := $(patsubst %.cc, %.o, $(wildcard *.cc)  $(wildcard ./src/*.cc)) 
DEPS := $(patsubst %.o, %.P, $(OBJS)) $(addsuffix .P, $(EXES))

define cxx_compile_with_dependency_creation
  $(COMPILE.cc) -MD -o $@ $<
  @sed -e 's|.*:|$*.o:|' <$*.d >$*.P
  @sed -e 's/.*://' -e 's/\\$$//' <$*.d | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >>$*.P
  @rm -f $*.d
endef

define cxx_link_rule
  $(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
endef

%.o: %.cc
	$(call cxx_compile_with_dependency_creation)

%.o: %.cxx
	$(call cxx_compile_with_dependency_creation)

%: %.o
	$(call cxx_link_rule)

.PHONY: all
all: $(EXES)

AdstExtractor: AdstExtractor.o $(OBJS)

.PHONY: clean
clean:
	- $(RM) -f $(OBJS) $(addsuffix .o, $(EXES)) $(EXES) $(DEPS)

-include $(DEPS)

.PHONY: release
release: CXXFLAGS := -ggdb3 -Wall -Wextra -Wtype-limits -std=c++17 -O3
release: $(EXES)

AdstExtractor: AdstExtractor.o $(OBJS)
