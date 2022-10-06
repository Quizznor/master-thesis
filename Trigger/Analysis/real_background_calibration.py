from scipy.optimize import curve_fit
from Binaries import *

peak_evolution = []
cut_threshold = 30
leading_bins = 20
trailing_bins = 49
multiplicity = 1
dead_time = trailing_bins
combine_files = 3

first_trigger = lambda x : np.argmax(x > cut_threshold)

def gauss(x, mu, A, sigma):

    return A * np.exp( -(x - mu)**2 / (2 * sigma**2) )

for station_name in ["peru", "jaco"]:

    try:

        for index in range(0, len(os.listdir(f"/cr/tempdata01/filip/iRODS/{station_name}/")), 3):

            h1, h2, h3 = [], [], []
            c1, c2, c3 = [], [], []
            peak, charge = [], []
            filenames = []

            for increment in range(combine_files):

                Buffer = RandomTrace(station_name, index + increment)

                print(f"{index + increment}/{Buffer.all_n_files} {Buffer.random_file}: {len(Buffer._these_traces)} traces" )

                filenames.append(Buffer.random_file)
                station = np.array(Buffer._these_traces)
                pmt1, pmt2, pmt3 = station[:,0], station[:,1], station[:,2]
                cut_mask = np.logical_and(np.any(pmt1 > cut_threshold, axis = 1), np.any(pmt2 > cut_threshold, axis = 1), np.any(pmt3 > cut_threshold, axis = 1))

                for (p1, p2, p3) in zip(pmt1[cut_mask], pmt2[cut_mask], pmt3[cut_mask]):

                    # STRONG COINCIDENCE
                    for step in range(min([first_trigger(p1), first_trigger(p2), first_trigger(p3)]), 2048):

                        trigger_list = [p1[step] > cut_threshold, p1[step] > cut_threshold, p1[step] > cut_threshold]

                        if trigger_list.count(True) >= multiplicity:

                            start, stop = max(0, step - leading_bins), min(2048, step + trailing_bins)
                            p1_data, p2_data, p3_data = p1[start : stop], p2[start : stop], p3[start : stop]
                            
                            h1.append(max(p1_data)), h2.append(max(p2_data)), h3.append(max(p3_data))
                            c1.append(sum(p1_data)), c2.append(sum(p2_data)), c3.append(sum(p3_data))

                            step += dead_time       # keep iterating
                            # break                   # stop after first hit
                
                    # # WEAK COINCIDENCE
                    # trigger_bin = np.argmax(p1 > cut_threshold)
                    # start = max(0, trigger_bin - leading_bins)
                    # stop = min(trigger_bin + trailing_bins, 2048)

                    # p1_data, p2_data, p3_data = p1[start : stop], p2[start : stop], p3[start : stop]

                    # if np.any(p2_data > cut_threshold) and np.any(p3_data > cut_threshold):

                    #     h1.append(func(p1_data)), h2.append(func(p2_data)), h3.append(func(p3_data))

            try:

                # Fit VEM Peak
                for j, histogram in enumerate([h1, h2, h3]):

                    n, bins = np.histogram(histogram,  bins = 250, range = (cut_threshold, 1200))
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])

                    x, y = bin_centers[17 : 80], n[17 : 80]
                    x_smooth = np.linspace(min(x), max(x), 100)

                    # popt, pcov = curve_fit(parabola, x, y, p0 = [x[np.argmax(y)], -10, max(y)], bounds = ([0, -np.inf, 0], [np.inf, 0, np.inf]) )
                    popt, pcov = curve_fit( gauss, x, y, p0 = [x[np.argmax(y)], sum(y), 10], bounds = ([50, 0, 0], [np.inf, np.inf, np.inf]) )
                    model_fit = gauss(x_smooth, *popt)
                    peak_estimate = popt[0]

                    if peak_estimate <= 120: raise SignalError

                    peak.append(peak_estimate)

                # Fit VEM Charge
                for j, histogram in enumerate([c1, c2, c3]):

                    n, bins = np.histogram(histogram,  bins = 250, range = (cut_threshold, 12000))
                    bin_centers = 0.5 * (bins[1:] + bins[:-1])

                    x, y = bin_centers[20 : 70], n[20 : 70]
                    x_smooth = np.linspace(min(x), max(x), 100)

                    # popt, pcov = curve_fit(parabola, x, y, p0 = [x[np.argmax(y)], -10, max(y)], bounds = ([0, -np.inf, 0], [np.inf, 0, np.inf]) )
                    popt, pcov = curve_fit( gauss, x, y, p0 = [x[np.argmax(y)], sum(y), 10], bounds = ([800, 0, 0], [np.inf, np.inf, np.inf]) )
                    model_fit = gauss(x_smooth, *popt)
                    charge_estimate = popt[0]

                    if charge_estimate <= 1000: raise SignalError

                    charge.append(charge_estimate)
                
                # Fitting was succesful
                with open(f"/cr/tempdata01/filip/iRODS/temp/{station_name}_offline_estimate.csv", "a") as file:
                    
                    for filename in filenames:
                        file.write(f"{filename}  {peak[0]:.2f} {peak[1]:.2f} {peak[2]:.2f}    {int(charge[0])}   {int(charge[1])}   {int(charge[2])}\n")
                                        

                print("\tFit results written to disk =)")

            except SignalError:
                print("\tFit didn't converge =(")
                continue

                # for increment in range(combine_files):

                #     filename = f"{station_name}/{station_name}_randoms{(index + increment).zfill(4)}.csv"
                    # os.system(f"mv /cr/tempdata01/filip/iRODS/{station_name}/{filename} /cr/tempdata01/filip/iRODS/faulty_estimate/{filename}")
                    # print(f"mv /cr/tempdata01/filip/iRODS/{station_name}/{filename} /cr/tempdata01/filip/iRODS/faulty_estimate/{filename}")
    
    except IndexError:
        continue
