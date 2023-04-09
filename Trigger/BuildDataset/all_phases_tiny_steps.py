from Binaries import *

q_peak_compatibility = np.array([163.235 for _ in range(3)])
p1 = EventGenerator("all", split = 1, real_background = False, apply_downsampling = True, random_phase = 0, seed = 42, sigma = 0, q_peak = q_peak_compatibility, window_step = 1)
p2 = EventGenerator("all", split = 1, real_background = False, apply_downsampling = True, random_phase = 1, seed = 42, sigma = 0, q_peak = q_peak_compatibility, window_step = 1)
p3 = EventGenerator("all", split = 1, real_background = False, apply_downsampling = True, random_phase = 2, seed = 42, sigma = 0, q_peak = q_peak_compatibility, window_step = 1)
n_showers = len(p1)

Trigger = HardwareClassifier()

save_path = "/cr/data01/filip/models/HardwareClassifier/ROC_curve/compatibility_all_phases_tiny_steps"

if not os.path.exists(save_path):
    os.makedirs(save_path)

start_time = perf_counter_ns()
save_file = \
    {
        "TP" : f"{save_path}/true_positives.csv",
        "TN" : f"{save_path}/true_negatives.csv",
        "FP" : f"{save_path}/false_positives.csv",
        "FN" : f"{save_path}/false_negatives.csv"
    }
        
# open all files only once, increases performance
with open(save_file["TP"], "w") as TP, \
        open(save_file["TN"], "w") as TN, \
        open(save_file["FP"], "w") as FP, \
        open(save_file["FN"], "w") as FN:

    for i in range(len(p1)):

        b1 = p1.__getitem__(i)
        b2 = p2.__getitem__(p2.find(p1.files[i]))
        b3 = p3.__getitem__(p3.find(p1.files[i]))

        progress_bar(i, n_showers, start_time)
        random_file = p1.files[i]
        random_file = f"{'/'.join(random_file.split('/')[-3:])}"

        for t1, t2, t3 in zip(b1, b2, b3):

            StationID = t1.StationID
            SPDistance = t1.SPDistance
            Energy = t1.Energy
            Zenith = t1.Zenith
            n_muons = t1.n_muons
            n_electrons = t1.n_electrons
            n_photons = t1.n_photons

            save_string = f"{random_file} {StationID} {SPDistance} {Energy} {Zenith} {n_muons} {n_electrons} {n_photons} "
            max_charge_integral = -np.inf

            for w1, w2, w3 in zip(t1, t2, t3):

                # we inherently care more about the situation were we trigger as
                # these events are seen by CDAS. Thus handle FP/TP more strictly

                label1, integral1 = p1.calculate_label(t1)
                label2, integral2 = p2.calculate_label(t2)
                label3, integral3 = p3.calculate_label(t3)

                assert (label := label1) == label2 == label3, "labels don't match"
                integral = integral1

                has_triggered = np.any([Trigger(w1), Trigger(w2), Trigger(w3)])

                # we have detected a false positive in the trace
                if label != "SIG" and has_triggered:
                    FP.write(save_string + f"{integral}\n")
                
                # we have detected a true positive, break loop
                elif label == "SIG" and has_triggered:
                    TP.write(save_string + f"{integral}\n")
                    break

                # we haven't seen anything, keep searching
                else:
                    if integral > max_charge_integral:
                        max_charge_integral = integral

                    # TODO: TN handling would go here

            # loop didn't break, we didn't see shit
            else:
                FN.write(save_string + f"{max_charge_integral}\n")

        if i + 1 >= n_showers: break        