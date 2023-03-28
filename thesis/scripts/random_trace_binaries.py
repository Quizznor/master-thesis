from Binaries import *

# For Random traces
# for station in ["lo_qui_don"]:
#     fft_container = np.zeros(2048)
#     n_files = len(os.listdir(f"/cr/tempdata01/filip/iRODS/{station}/"))

#     for i in range(n_files):
#         RandomTraces = RandomTrace(station = station, index = i)

#         for j, example in enumerate(RandomTraces._these_traces):
#             for pmt in example:
#                 fft_container += abs(np.fft.fft(pmt))**2  / (3 * len(RandomTraces._these_traces) * n_files)

#     np.savetxt(f"/cr/users/filip/thesis/scripts/random-traces-fft/{station}.csv", fft_container)


Showers = EventGenerator("all", real_background = False, split = 1)

all_fft = np.zeros(2048)
start = perf_counter_ns()

for i, batch in enumerate(Showers):

    progress_bar(i, Showers.__len__(), start)

    per_batch_avg = np.zeros(2048)
    batch_size = len(batch)

    for station in batch:

        fft = np.fft.fft(np.mean(station.Signal, axis = 0))
        per_batch_avg += abs(fft)**2 / batch_size

    all_fft += per_batch_avg / Showers.__len__()

np.savetxt(f"/cr/users/filip/thesis/scripts/random-traces-fft/showers.csv", all_fft)