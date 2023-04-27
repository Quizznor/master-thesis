#!/usr/bin/python3

from Binaries import *

def do_permutation(pmt1, pmt2, pmt3, key):
    if key == 0: return pmt1, pmt2, pmt3                # perm 0 : [1, 2, 3]
    elif key == 1: return pmt1, pmt3, pmt2              # perm 1 : [1, 3, 2]
    elif key == 2: return pmt2, pmt1, pmt3              # perm 2 : [2, 1, 3]
    elif key == 3: return pmt2, pmt3, pmt1              # perm 3 : [2, 3, 1]
    elif key == 4: return pmt3, pmt1, pmt2              # perm 4 : [3, 1, 2]
    elif key == 5: return pmt3, pmt2, pmt1              # perm 5 : [3, 2, 1]

permutation = int(sys.argv[1])
save_dir = f"permutation_{permutation}"

LSTM = Ensemble("120_LSTM_3L_Downsampled_AllEnergies_5_0VEM", supress_print = True)
BestModel = LSTM.get_best_model("final_predictions")
save_path = "/cr/data01/filip/models/" + BestModel.name + f"/model_{BestModel.epochs}/ROC_curve/" + save_dir
Events = EventGenerator([], apply_downsampling = True, q_peak = np.array([GLOBAL.q_peak_compatibility for _ in range(3)]), ignore_low_vem = 5.0, split = 1)      # cut goes here
Events.files += BestModel.get_files("validation")

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

    for batch_no, traces in enumerate(Events):

        progress_bar(batch_no, len(Events), start_time)
        random_file = Events.files[batch_no]
        random_file = f"{'/'.join(random_file.split('/')[-3:])}"

        for trace in traces:

            pmt1, pmt2, pmt3 = trace.pmt_1, trace.pmt_2, trace.pmt_3
            int1, int2, int3 = trace.int_1, trace.pmt_2, trace.int_3

            trace.pmt_1, trace.pmt_2, trace.pmt_3 = do_permutation(pmt1, pmt2, pmt3, permutation)
            trace.int_1, trace.int_2, trace.int_3 = do_permutation(int1, int2, int3, permutation)

            StationID = trace.StationID
            SPDistance = trace.SPDistance
            Energy = trace.Energy
            Zenith = trace.Zenith
            n_muons = trace.n_muons
            n_electrons = trace.n_electrons
            n_photons = trace.n_photons

            save_string = f"{random_file} {StationID} {SPDistance} {Energy} {Zenith} {n_muons} {n_electrons} {n_photons} "
            max_charge_integral = -np.inf

            for window in trace:

                # we inherently care more about the situation were we trigger as
                # these events are seen by CDAS. Thus handle FP/TP more strictly

                label, integral = Events.calculate_label(trace)
                has_triggered = BestModel.__call__(window)

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

        if batch_no + 1 >= len(Events): break