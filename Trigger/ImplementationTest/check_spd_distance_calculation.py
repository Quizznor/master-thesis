from Binaries import *

def do_t3_simulation(n_points):

    fitparams = np.loadtxt("/cr/users/filip/Trigger/ImplementationTest/fit_param_example.csv")
    plt.rcParams["figure.figsize"] = [25, 18]
    plt.rcParams["font.size"] = 22
    ax = plt.figure().add_subplot(projection = '3d')

    # set up plot
    # ax.text( 635, -55, 0, "T3 detected", fontsize = 22, zdir='z')
    # ax.text(1395, 775, 0, "T3 missed", fontsize = 22, zdir='z')
    symmetry_line = lambda x : 1500 - x
    X = np.linspace(700, 1550, 100)
    ax.scatter([0, 1500, 750, 2250], [0, 0, 750, 750], s = 100, c = "k")
    ax.plot(X, symmetry_line(X), ls = ":", c = "k", zorder = 0, lw = 2)
    # ax.add_patch(Polygon([[0,0, 0], [1500, 0, 0], [750, 750, 0]], closed = True, color = "green", alpha = 0.1, lw = 0))
    # ax.add_patch(Polygon([[750, 750, 0], [1500, 0, 0], [2250, 750, 0]], closed = True, color = "red", alpha = 0.1, lw = 0))

    # create shower cores in target area
    theta_bins = [0, 26, 38, 49, 60, 90]
    ys = np.random.uniform(0, 750, n_points)
    xs = np.random.uniform(0, 1500, n_points) + ys
    reflect = [ ys[i] > symmetry_line(xs[i]) for i in range(len(xs))]
    xs[reflect] = -xs[reflect] + 2250
    ys[reflect] = -ys[reflect] + 750

    # do the T3 simulation
    t3_hits, t3_misses = np.zeros((7, 5)), np.zeros((7, 5))
    stations = [[0, 0, 0], [1500, 0, 0], [750, 750, 0]]

    for x, y in zip(xs, ys):
        energy_and_theta = np.random.randint(0, len(fitparams))
        energy, t = energy_and_theta // 5, energy_and_theta % 5
        fit_function = lambda spd : station_hit_probability(x, *fitparams[energy_and_theta])

        # choose theta, phi at random, calculate shower_plane_distance
        theta = np.radians(np.random.uniform(theta_bins[t], theta_bins[t + 1]))
        phi = np.radians(np.random.uniform(0, 360))
        phi, theta = 0, np.radians(50)
        sp_distances = []

        core_position = np.array([x, y, 0])
        core_origin = 1000 * np.sin(theta) * np.array([np.cos(phi), np.sin(phi), 1/np.tan(theta)]) + core_position
        core_origin_projection = core_origin[:2] + [0]

        ax.scatter(*core_position, label = "core", s = 100)
        ax.scatter(*core_origin, label = "origin", s = 100)
        ax.scatter(*core_origin_projection, s = 10)
        
        for station in stations:


            d = np.divide(core_origin - core_position, np.linalg.norm(core_origin - core_position))

            s = np.dot(core_position - station, d)
            t = np.dot(station - core_origin, d)
            h = np.maximum.reduce([s, t, 0])
            c = np.cross(station - core_position, d)
            sp_distances.append(np.hypot(h, np.linalg.norm(c)))

            # core_vector_start = np.array(station)
            # core_vector_end = core_vector_start + np.sin(theta) * np.array([np.cos(phi), np.sin(phi), 1/np.tan(theta)])

            # # normed shower axis
            # core_axis = np.divide(core_vector_end - core_vector_start, np.linalg.norm(core_vector_start - core_vector_end))
            
            # # signed parallel distance components
            # s = np.dot(core_vector_start - station, core_axis)
            # t = np.dot(station - core_vector_end, core_axis)
            # h = np.maximum.reduce([s, t, 0])
            # c = np.cross(station - core_vector_start, core_axis)
            # sp_distance = np.hypot(h, np.linalg.norm(c))

            # print(sp_distance)

        print(np.degrees(theta), np.degrees(phi))

        distances = [np.linalg.norm(station - np.array([x, y, 0])) for station in stations]

        # # #  In case of paranoia regarding distance calculation break comment
        # ax.add_patch(plt.Circle((0, 0), sp_distances[0], color='b', fill=False))
        # ax.add_patch(plt.Circle((1500, 0), sp_distances[1], color='b', fill=False))
        # ax.add_patch(plt.Circle((750, 750), sp_distances[2], color='b', fill=False))

        trigger_probabilities = [fit_function(distance) for distance in distances]
        dice_roll = np.random.uniform(0, 1, 3)
        
        # plt.scatter(x, y, c = "k", s = 90)

        # if np.all(dice_roll < trigger_probabilities):
        #     t3_hits[energy][theta] += 1
        #     plt.scatter(x, y, c = "k")
        # else:
        #     x, y = 2250 - x, 750 -y
        #     t3_misses[energy][theta] += 1
        #     plt.scatter(x, y, c = "r")

    # ax.set_aspect('equal')
    # ax.view_init(azim=np.degrees(theta), elev=90)
    plt.xlabel("Easting / m")
    plt.ylabel("Northing / m")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    do_t3_simulation(1)