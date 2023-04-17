#!/usr/bin/python3

from Binaries import *
ensemble_no = int(sys.argv[1]) + 1

ThisNN = NNClassifier(f"ENSEMBLES/120_TwoLayer_FullBandwidth_HighEnergies_8Muon/ensemble_{ensemble_no:02}")

# calculate trigger rate on random traces
print(f"\ncalculating trigger rate on 0.5s (~30 000 Traces) of random traces now")
f, df, n, n_trig, t = ThisNN.production_test(30000, apply_downsampling = False)

with open(f"/cr/data01/filip/models/{ThisNN.name}/model_{ThisNN.epochs}/production_test.csv", "w") as random_file:
    random_file.write("# f  df  n_traces  n_total_triggered  total_trace_duration\n")
    random_file.write(f"{f} {df} {n} {n_trig} {t}")

# AllEvents = EventGenerator(":19_19.5", split = 1)
# AllEvents.files += ThisNN.get_files("validation")
# ThisNN.make_signal_dataset(AllEvents, "all_energies_no_cuts")