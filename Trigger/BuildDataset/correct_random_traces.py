from TriggerStudyBinaries_v6 import *
import os, sys, logging

class ElectronicsError(Exception) : pass
class SignalError(Exception) : pass

# Implementation of Tobias Schulz baseline algorithm in python
def corrected_baseline(trace : np.ndarray) -> np.ndarray :

    def calculate_baseline(*args) -> list :

        baselines = []
        for window in args:

            while True:

                mode, count = None, 0
                
                # calculate mode of the trace windows
                for value in range(int(min(window)), int(max(window))):
                    current_count = window.count(value)
                    if current_count > count:
                        mode, count = value, current_count

                upper, lower = mode + 2* np.std(window), mode - 2* np.std(window)
                to_remove = [bin for bin in window if bin > upper or bin < lower]           
                if to_remove != []: [window.remove(value) for value in to_remove]
                else: break

            baselines.append([np.mean(window), np.std(window)])

        return baselines

    (i, sigma_i), (f, sigma_f) = calculate_baseline(list(trace[:300]), list(trace[-300:]))
    delta_B, sigma_B = f - i, np.sqrt(sigma_i**2 + sigma_f**2)

    if delta_B >= 5 * sigma_B: raise ElectronicsError                               # discard trace
    
    elif 5 * sigma_B > delta_B >= 0:                                                # approx. constant
        
        Baseline, _ =  calculate_baseline(list(trace))[0]
        return trace - Baseline

    elif 0 > delta_B >= -sigma_B or (-sigma_B > delta_B and (max(trace) - i) < 50): # step function

        front, end = trace[:np.argmax(trace)], trace[np.argmax(trace):]
        return np.array(list(front - i) + list(end - f))
    
    elif -sigma_B > delta_B:                                                        # signal undershoot

        raise SignalError
        # TODO
        # interpolated_baseline = [i - delta_B * (bin - 300)/(2048 - 600) for bin in range(300, 2048 - 300)]
        # for iteration in range(5): pass
        


if __name__ == "__main__":

    save_path = "/cr/tempdata01/filip/iRODS/Background/"
    all_randoms = os.listdir(save_path)

    for a, random_trace in enumerate(all_randoms, 1):

        # os.system(f"touch /cr/tempdata01/filip/iRODS/corrected/{random_trace}")
        data = np.loadtxt(save_path + random_trace)
        data = np.split(data, len(data) // 3)

        for b, station in enumerate(data, 1):

            print(f"Fetching file {a}/{len(all_randoms)} - station {b}/{len(data)}", end = "\r")

            corrected_station = []
            
            try:
                for trace in station:

                    FPGA_baseline, trace = trace[0], trace[1:] + trace[0]
                    corrected_station.append(corrected_baseline(trace))

                with open(f"/cr/tempdata01/filip/iRODS/corrected/{random_trace}", "a") as corrected_file:
                    np.savetxt(corrected_file, np.floor(np.array(corrected_station)), fmt = "%i")
                    
            except SignalError: 
                with open(f"/cr/tempdata01/filip/iRODS/SignalError/{random_trace}", "a") as file:
                    np.savetxt(file, station, fmt = "%i")
            except ElectronicsError:
                with open(f"/cr/tempdata01/filip/iRODS/ElectronicsError/{random_trace}", "a") as file:
                    np.savetxt(file, station, fmt = "%i")
            except TypeError:
                with open(f"/cr/tempdata01/filip/iRODS/TypeError/{random_trace}", "a") as file:
                    np.savetxt(file, station, fmt = "%i")

            # #########################################################################
            # test = np.loadtxt(f"/cr/tempdata01/filip/iRODS/corrected/{random_trace}")

            # plt.figure()
            # plt.title("Baseline corrected")
            # for pmt in test:
            #     plt.plot(range(len(pmt)), pmt, label = f"mean = {np.mean(pmt)}")

            # plt.legend()

            # plt.figure()
            # plt.title("FPGA estimate")
            # for pmt in station:
            #     plt.plot(range(len(pmt) - 1), pmt[1:], label = f"mean = {np.mean(pmt[1:])}")

            # plt.legend()
            # plt.show()

            # raise StopIteration
            # #########################################################################
