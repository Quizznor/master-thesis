from scipy.optimize import curve_fit
from Binaries import *

peak_evolution = []
cut_threshold = 30
leading_bins = 20
trailing_bins = 49
multiplicity = 1
dead_time = trailing_bins
combine_files = 30
skip_files = 30

first_trigger = lambda x : np.argmax(x > cut_threshold)

def gauss(x, mu, A, sigma):

    return A * np.exp( -(x - mu)**2 / (2 * sigma**2) )

def background_vem(x, A, c): return A - (x - 50) * c
def background_charge(x, A, c): return A - (x - 800) * c

for station_name in ["jaco"]:    # ["peru", "jaco", "nuria"]:

    try:

        for index in range(0, len(os.listdir(f"/cr/tempdata01/filip/iRODS/{station_name}/")), skip_files):

            h1, h2, h3 = [], [], []
            c1, c2, c3 = [], [], []
            peak, charge = [], []
            peak_error, charge_error = [], []
            filenames = []

            for increment in range(combine_files):

                Buffer = RandomTrace(station_name, index + increment)

                print(f"{index + increment}/{Buffer.all_n_files} {Buffer.random_file}: {len(Buffer._these_traces)} traces" )

                filenames.append(Buffer.random_file)
                station = np.array(Buffer._these_traces)
                pmt1, pmt2, pmt3 = station[:,0], station[:,1], station[:,2]
                cut_mask = np.logical_and(np.any(pmt1 > cut_threshold, axis = 1), np.any(pmt2 > cut_threshold, axis = 1), np.any(pmt3 > cut_threshold, axis = 1))

                for (p1, p2, p3) in zip(pmt1[cut_mask], pmt2[cut_mask], pmt3[cut_mask]):

                    steps = iter(range(min([first_trigger(p1), first_trigger(p2), first_trigger(p3)]), 2048))

                    # STRONG COINCIDENCE
                    for step in steps:

                        trigger_list = [p1[step] > cut_threshold, p2[step] > cut_threshold, p3[step] > cut_threshold]

                        if trigger_list.count(True) >= multiplicity:

                            start, stop = max(0, step - leading_bins), min(2048, step + trailing_bins)
                            p1_data, p2_data, p3_data = p1[start : stop], p2[start : stop], p3[start : stop]
                            
                            h1.append(max(p1_data)), h2.append(max(p2_data)), h3.append(max(p3_data))
                            c1.append(sum(p1_data)), c2.append(sum(p2_data)), c3.append(sum(p3_data))

                            for _ in range(dead_time): next(steps, None)

            # Fit VEM Peak
            for j, histogram in enumerate([h1, h2, h3]):

                n, bins = np.histogram(histogram,  bins = 250, range = (cut_threshold, 1200))
                bin_centers = 0.5 * (bins[1:] + bins[:-1])

                # Signal region
                x, y = bin_centers[20 : 100], n[20 : 100]
                x_smooth = np.linspace(min(x), max(x), 100)

                x_out = list(bin_centers[0 : 20]) + list(bin_centers[100 : -1])
                y_out = list(n[0 : 20]) + list(n[100 : -1])
                x_out_smooth = np.linspace(min(x_out), max(x_out), 100)

                # fitting background exponential first
                y_out_log = np.log(y_out)
                y_out_log[y_out_log == -np.inf] = 0
                popt_background, pcov_background = curve_fit(background_vem, x_out, y_out_log, p0 = [y_out_log[0], 0.2], bounds = ([0, 0], [np.inf, np.inf]))
                background_fit = np.exp(background_vem(x_out_smooth, *popt_background))

                # fitting signal region with substracted background
                popt, pcov = curve_fit(gauss, x, y - np.exp(background_vem(x, *popt_background)), p0 = [GLOBAL.q_peak, 50, 10], bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]) )
                model_fit = gauss(x_smooth, *popt) + np.exp(background_vem(x_smooth, *popt_background))
                peak_error_estimate = np.sqrt(pcov[0][0])
                peak_estimate = popt[0]

                # if peak_estimate <= 120: raise SignalError

                peak.append(peak_estimate)
                peak_error.append(peak_error_estimate)

            # Fit VEM Charge
            for j, histogram in enumerate([c1, c2, c3]):

                n, bins = np.histogram(histogram,  bins = 250, range = (cut_threshold, 12000))
                bin_centers = 0.5 * (bins[1:] + bins[:-1])

                # Signal region
                x, y = bin_centers[20 : 100], n[20 : 100]
                x_smooth = np.linspace(min(x), max(x), 100)

                x_out = list(bin_centers[0 : 20]) + list(bin_centers[100 : -1])
                y_out = list(n[0 : 20]) + list(n[100 : -1])
                x_out_smooth = np.linspace(min(x_out), max(x_out), 100)

                # fitting background exponential first
                y_out_log = np.log(y_out)
                y_out_log[y_out_log == -np.inf] = 0
                popt_background, pcov_background = curve_fit(background_charge, x_out, y_out_log, p0 = [y_out_log[0], 0.2], bounds = ([0, 0], [np.inf, np.inf]))
                background_fit = np.exp(background_charge(x_out_smooth, *popt_background))

                # fitting signal region with substracted background
                popt, pcov = curve_fit(gauss, x, y - np.exp(background_charge(x, *popt_background)), p0 = [GLOBAL.q_charge, 800, 200], bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]) )
                model_fit = gauss(x_smooth, *popt) + np.exp(background_charge(x_smooth, *popt_background))
                charge_error_estimate = np.sqrt(pcov[0][0])
                charge_estimate = popt[0]


                charge.append(charge_estimate)
                charge_error.append(charge_error_estimate)
            
            # Fitting was succesful
            with open(f"/cr/tempdata01/filip/iRODS/temp/{station_name}_offline_estimate_background_substracted.csv", "a") as file:
                
                for filename in filenames:
                    file.write(f"{filename}  {peak[0]:.2f} {peak[1]:.2f} {peak[2]:.2f} ")
                    file.write(f"{peak_error[0]:.2e} {peak_error[1]:.2e} {peak_error[2]:.2e} ")
                    file.write(f"{int(charge[0])} {int(charge[1])} {int(charge[2])} ")
                    file.write(f"{int(charge_error[0])}   {int(charge_error[1])}   {int(charge_error[2])}\n")                      

            print("\tFit results written to disk =)")

    except ValueError:
        continue
