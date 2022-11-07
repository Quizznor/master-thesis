from Binaries import *

triggers = np.loadtxt("/cr/tempdata01/filip/iRODS/temp/triggers.csv", dtype = str)

stations, counts = np.unique(triggers, return_counts = True)
stations_int = [int(station[-8:-4]) for station in stations]
single_trace_duration = GLOBAL.single_bin_duration * GLOBAL.n_bins
rates, sigma_rates = [], []

for i, station in enumerate(stations_int):
    Buffer = RandomTrace(index = station - 1)
    all_traces = len(Buffer._these_traces)
    all_traces_duration = all_traces * single_trace_duration
    rate_error = np.sqrt(counts[i]) / all_traces_duration
    rate = counts[i] / all_traces_duration

    print(stations[i], counts[i], all_traces, rate, rate_error)

    with open("/cr/tempdata01/filip/iRODS/MonitoringData/trigger_rates.csv", "a") as save_file:
        save_file.write(f"{stations[i]} {counts[i]} {all_traces} {rate:.6} {rate_error:.9}\n")