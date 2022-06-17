# GPLv2 and 2C-BSD
# Copyright (c) Darko Veberic, 2014

ifndef ROOTSYS
  ROOTSYS := /usr
endif

CPPFLAGS := $(shell $(ROOTSYS)/bin/root-config --cflags) -std=c++14 -I. -include sstream
#CXXFLAGS := -Wall -Wextra -g -ggdb3 -O2 -fno-inline -pipe #-D_GLIBCXX_DEBUG
#CXXFLAGS := -Wall -Wextra -Wtype-limits -std=c++11 -ggdb3 -O2 -fno-inline -fsanitize=address -pipe
#CXXFLAGS := -Wall -Wextra -Wtype-limits -std=c++11 -ggdb3 -O0 -fno-inline -pipe # valgrind
CXXFLAGS := -Wall -Wextra -pipe -O3 #-g -ggdb3
LDFLAGS := -lboost_program_options $(shell $(ROOTSYS)/bin/root-config --ldflags --libs) -lMinuit -lboost_iostreams

SHELL := bash
SUFFIXES := .o .cc .cxx

VPATH := . io interface sd utl STAnalysis T2Dump t2iosd t2 t3 CorrelationAnalysis monit traces LightningUtilities

EXES := $(basename $(wildcard *.cxx))
OBJS := $(foreach d, $(VPATH), $(patsubst %.cc, %.o, $(wildcard $(d)/*.cc)))

LINKDEFS := $(foreach d, $(VPATH), $(patsubst %.LinkDef.h, %, $(wildcard $(d)/*.LinkDef.h)))
OBJS += $(addsuffix .Dict.o, $(LINKDEFS))
.PRECIOUS: $(addsuffix .Dict.h, $(LINKDEFS)) $(addsuffix .Dict.cc, $(LINKDEFS))

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

%.Dict.cc %.Dict.h: %.h %.LinkDef.h
	$(ROOTSYS)/bin/rootcling -f $@ -I. $^
	- cp $(dir $*)*.Dict_rdict.pcm .
#	$(ROOTSYS)/bin/rootcint -v4 -cint -f $@ -c -p -I. $^

.PHONY: all
all: $(EXES)
MC_T2StabilityTester: MC_T2StabilityTester.o $(OBJS)
T2DumpTest: T2DumpTest.o $(OBJS)
MCT2s: MCT2s.o $(OBJS)
JoinCandidateWithMC: JoinCandidateWithMC.o $(OBJS)
ExtractOfflineCandidates: ExtractOfflineCandidates.o $(OBJS)

count_active_stations: count_active_stations.o $(OBJS)
add_nt2_summaries: add_nt2_summaries.o $(OBJS)
t2dump_part_to_json: t2dump_part_to_json.o $(OBJS)
doublet_rate_finder: doublet_rate_finder.o $(OBJS)

extract_clustert2_multiplicity: extract_clustert2_multiplicity.o $(OBJS)

dump_trigger_rates: dump_trigger_rates.o $(OBJS)

single_station_second_excess_finding: single_station_second_excess_finding.o $(OBJS)
dump_hgsat_triggers: dump_hgsat_triggers.o $(OBJS)
dump_long_signal_flags: dump_long_signal_flags.o $(OBJS)
compact_analysis: compact_analysis.o $(OBJS)
compact_time_analysis: compact_time_analysis.o $(OBJS)
adst_event_time_analysis: adst_event_time_analysis.o $(OBJS)
adst_showerplane_time_differences: adst_showerplane_time_differences.o $(OBJS)
create_adst_dt_histograms: create_adst_dt_histograms.o $(OBJS)

MergeT2DumpT3Data: MergeT2DumpT3Data.o $(OBJS)
JoinedBrowser: JoinedBrowser.o $(OBJS)

SelectEvents: SelectEvents.o $(OBJS)
T2DumpAnalysis: T2DumpAnalysis.o $(OBJS)
T3Simulator: T3Simulator.o $(OBJS)
FitPointSources: FitPointSources.o $(OBJS)
MatchCandidatesMC: MatchCandidatesMC.o $(OBJS)
T2Browser: T2Browser.o $(OBJS)
CompareToBg: CompareToBg.o $(OBJS)
ShiftTimes: ShiftTimes.o $(OBJS)
T3Comparison: T3Comparison.o $(OBJS)
DumpT2Data: DumpT2Data.o $(OBJS)
ComparePattern: ComparePattern.o $(OBJS)
VideoMaker: VideoMaker.o $(OBJS)
ReBinCandidates: ReBinCandidates.o $(OBJS)
DeltaTScan: DeltaTScan.o $(OBJS)

NTriggerDeltaT: NTriggerDeltaT.o $(OBJS)
dump_trigger_tuples: dump_trigger_tuples.o $(OBJS)
dump_station_triggers: dump_station_triggers.o $(OBJS)
create_trigger_ntuples: create_trigger_ntuples.o $(OBJS)
build_quadruplets: build_quadruplets.o $(OBJS)
reject_bad_deltaT_stations: reject_bad_deltaT_stations.o $(OBJS)
reject_bad_histograms: reject_bad_histograms.o $(OBJS)
process_mergedcorrelationevents: process_mergedcorrelationevents.o $(OBJS)
process_stationcorrelationevents: process_stationcorrelationevents.o $(OBJS)

SelectLightningHours: SelectLightningHours.o $(OBJS)
TestCandidates: TestCandidates.o $(OBJS)
station_trigger_history_finder: station_trigger_history_finder.o $(OBJS)
join_iosd_t2dumps: join_iosd_t2dumps.o $(OBJS)
JoinedHistoryBrowser: JoinedHistoryBrowser.o $(OBJS)
process_joined_history: process_joined_history.o $(OBJS)

reconstruct_lightning_data: reconstruct_lightning_data.o $(OBJS)
GLD_to_correlation_events: GLD_to_correlation_events.o $(OBJS)
GLD360_root_to_asci: GLD360_root_to_asci.o $(OBJS)

correlation_analysis: correlation_analysis.o $(OBJS)
station_correlation_analysis: station_correlation_analysis.o $(OBJS)
join_correlationpoints_events: join_correlationpoints_events.o $(OBJS)
correlation_analysis_2: correlation_analysis_2.o $(OBJS)
join_correlationpoints_events: join_correlationpoints_events.o $(OBJS)
read_lightning_data: read_lightning_data.o $(OBJS)
convert_single_station_lightning_data: convert_single_station_lightning_data.o $(OBJS)
add_mc_events: add_mc_events.o $(OBJS)
generate_mc_events: generate_mc_events.o $(OBJS)

build_station_status: build_station_status.o $(OBJS)
create_station_status_plots: create_station_status_plots.o $(OBJS)

correlate_lightning_strikes: correlate_lightning_strikes.o $(OBJS)
find_correlations: find_correlations.o $(OBJS)
calibrate_uniformity_tests: calibrate_uniformity_tests.o $(OBJS)

wide_delta_t: wide_delta_t.o $(OBJS)

test_t2dumpfile_write: test_t2dumpfile_write.o $(OBJS)
test_correlation_point_selector: test_correlation_point_selector.o $(OBJS)
test_BinnedScan: test_BinnedScan.o $(OBJS)
test_correlation_event: test_correlation_event.o $(OBJS)

test_StationStatus: test_StationStatus.o $(OBJS)
test_RejectionManager: test_RejectionManager.o $(OBJS)

test_iosd_stationtime: test_iosd_stationtime.o $(OBJS)
test_t2iosd_stationdata: test_t2iosd_stationdata.o $(OBJS)
test_t2iosd_eventtime: test_t2iosd_eventtime.o $(OBJS)
test_t2iosd_eventdata: test_t2iosd_eventdata.o $(OBJS)

test_2D_geometry: test_2D_geometry.o $(OBJS)
test_merged_lightning_event: test_merged_lightning_event.o $(OBJS)

.PHONY: check
check: $(basename $(wildcard test_*.cxx))
	for t in $^ ; do echo check: run $$t ; ./$$t || exit $$? ; done

.PHONY: clean
clean:
	- $(RM) $(OBJS) spe/*.Dict.{h,cc} $(addsuffix .o, $(EXES)) $(EXES) $(DEPS)
	- $(RM) $(foreach d, $(VPATH), $(wildcard $(d)/*.Dict.h) $(wildcard $(d)/*.Dict.cc) $(wildcard $(d)/*.Dict_rdict.pcm))

-include $(DEPS)
