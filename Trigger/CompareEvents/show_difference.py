from Binaries import *

q_peak = np.array([1, 1, 1])
q_charge = np.array([GLOBAL.q_charge / GLOBAL.q_peak for _ in range(3)])

EventsVEMdownsampled = EventGenerator(["19_19.5"], real_background = False, sigma = 0, split = 1, q_peak = q_peak, q_charge = q_charge, apply_downsampling = True)
EventsVEMdownsampled.files = ["/cr/tempdata01/filip/QGSJET-II/COMPARE/VEM/" + file for file in os.listdir("/cr/tempdata01/filip/QGSJET-II/COMPARE/VEM")]

EventsVEMdownsampled.physics_test(save_dir = "/cr/users/filip/plots/physics_test_VEM_downsampled.png")