from Binaries import *

p1 = EventGenerator("all", split = 1, real_background = True, apply_downsampling = True, random_phase = 0, seed = 42)
p2 = EventGenerator("all", split = 1, real_background = True, apply_downsampling = True, random_phase = 1, seed = 42)
p3 = EventGenerator("all", split = 1, real_background = True, apply_downsampling = True, random_phase = 2, seed = 42)
n_showers = len(p1)

Trigger = HardwareClassifier()

save_path = "/cr/data01/filip/models/HardwareClassifier/ROC_curve/compatibility_all_phases"

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

    for batch_no, (b1, b2, b3) in enumerate(zip(p1, p2, p3)):

        progress_bar(batch_no, n_showers, start_time)
        random_file = p1.files[batch_no]
        random_file = f"{'/'.join(random_file.split('/')[-3:])}"

        for t1, t2, t3 in zip(b1, b2, b3):

            StationID = t1.StationID
            SPDistance = t1.SPDistance
            Energy = t1.Energy
            Zenith = t1.Zenith
            n_muons = t1.n_muons
            n_electrons = t1.n_electrons
            n_photons = t1.n_photons

            assert StationID == t2.StationID == t3.StationID, "Station IDs do not match"
            assert SPDistance == t2.SPDistance == t3.SPDistance, "SP Distances do not match"
            assert Energy == t2.Energy == t3.Energy, "Energies do not match"
            assert Zenith == t2.Zenith == t3.Zenith, "Zeniths do not match"
            assert n_muons == t2.n_muons == t3.n_muons, "n_mouns do not match"
            assert n_electrons == t2.n_electrons == t3.n_electrons, "n_electrons do not match"
            assert n_photons == t2.n_photons == t3.n_photons, "n_photons do not match"

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

        if batch_no + 1 >= n_showers: break        