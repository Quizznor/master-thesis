from Binaries import *

for station in ["nuria", "lo_qui_don"]:
    fft_container = np.zeros(2048)
    n_files = len(os.listdir(f"/cr/tempdata01/filip/iRODS/{station}/"))

    for i in range(n_files):
        RandomTraces = RandomTrace(station = station, index = i)

        for j, example in enumerate(RandomTraces._these_traces):
            for pmt in example:
                fft_container += abs(np.fft.fft(pmt)) / (3 * len(RandomTraces._these_traces) * n_files)

    np.savetxt(f"/cr/users/filip/thesis/scripts/random-traces-fft/{station}.csv", fft_container)