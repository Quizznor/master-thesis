from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *
from .Hardware import *
from .Network import *
from .Ensemble import *

# plot the estimated confidence range of provided classifiers
def confidence_comparison(confidence_level, *args, **kwargs):

    raise NotImplementedError("Calculation of arbitrary p_x needs to be altered to reflect new LTP fitfunc")

    y_max = kwargs.get("ymax", 2500)
    labels = kwargs.get("labels", None)
    energy_labels = ["16_16.5", "16.5_17", "17_17.5", "17.5_18", "18_18.5", "18.5_19", "19_19.5"]
    theta_labels = [r"$0^\circ$", r"$33^\circ$", r"$44^\circ$", r"$51^\circ$", r"$56^\circ$", r"$65^\circ$"]
    colors = ["steelblue", "orange", "green", "maroon", "lime", "indigo", "slategray"]

    try:
        if labels and len(labels) != len(args): raise ValueError
    except:
        sys.exit("Provided labels doesn't match the provided fit parameters")

    fig, axes = plt.subplots(nrows = len(theta_labels) - 1, sharex = True, sharey = True)
    axes[0].set_title(f"Trigger characteristics for r$_{{{confidence_level * 1e2:.0f}}}$")

    for i, fit_params in enumerate(args):
            
            acc, p50, scale = fit_params
        
            station_trigger_probability = lambda x : station_hit_probability(x, acc, p50, scale)
            inverse_trigger_probability = lambda y : p50 - np.log(acc/(1-y) - 1) / scale

            # calculate gradient
            exp = lambda x, k, b : np.exp(-k * (x - b))
            d_accuracy = station_trigger_probability(confidence_level) / acc
            d_p50 = acc * scale * exp(confidence_level, scale, p50) / (1 + exp(confidence_level, scale, p50))**2
            d_scale = acc * (p50 - confidence_level) * exp(confidence_level, scale, p50) / (1 + exp(confidence_level, scale, p50))**2
            grad = np.array([d_accuracy, d_p50, d_scale])

            axes[t].errorbar(e, inverse_trigger_probability(confidence_level), xerr = 0.5, capsize = 3, c = colors[i], elinewidth = 1, fmt = "s")

    axes[0].set_xticks(range(7), energy_labels)

    fig.text(0.5, 0.04, 'Energy range', ha='center', fontsize = 27)
    fig.text(0.04, 0.5, 'Detection radius / m', va='center', rotation='vertical', fontsize = 27)
    
    for i, ax in enumerate(axes):
        if labels: 
            for ii, label in enumerate(labels):
                ax.scatter([], [], marker = "s", c = colors[ii], label = labels[ii])

        ax.legend(title = theta_labels[i] + r"$\leq$ $\theta$ < " + theta_labels[i + 1])
        ax.axhline(0, c = "gray", ls = ":", lw = 2)

    plt.ylim(-100, y_max)

class MoneyPlot():
     
    def __init__(self, name : str) -> None : 

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(name)
        self.ax.set_yscale("log")
        self.ax.set_ylabel("Random trace trigger rate / $\mathrm{Hz}$")
        self.ax.set_xlabel("Trigger efficiency")
        HardwareClassifier.plot_performance(self.ax)

        self.ax.errorbar([], [], c = "k", markersize = 15, mfc = "w", fmt = "-o", ls = "solid", lw = 2, label = "Classical triggers")
        self.ax.plot([], [], c = "k", ls = "--", label = "Th-T2 only")
        self.ax.plot([], [], c = "k", ls = ":", label = "ToT + ToTd combined")
        

        self.buffer_x, self.buffer_y = [], []

    def add(self, ensemble : str, dataset, **kwargs) -> None :

        try:
            acc, acc_err, rate, rate_err = np.loadtxt(f"/cr/users/filip/MoneyPlot/data/{ensemble}/{dataset}.csv", unpack = True)

            if len(acc) != 10: print(f"[WARN] -- Incomplete predictions for {ensemble}: {dataset}... You may want to recalculate this")

            n_rate, bins = np.histogram(rate, bins = 10)
            current_y, scaling = min(rate), 0.005
            bin_width = bins[1] - bins[0]
            boxes = []

            for i, rate_bin in enumerate(n_rate):
                boxes.append(Rectangle((np.mean(acc) - scaling * rate_bin, current_y), width = 2*scaling*rate_bin, height = bin_width))
                current_y += bin_width
            
            x, y = np.mean(acc), np.mean(rate)
            color = kwargs.get("color", None)
            label = kwargs.get("label", None)
            self.ax.add_collection(PatchCollection(boxes, facecolor = "slategray", lw = 0, alpha = 0.2))
            self.ax.errorbar(x, y, xerr = np.std(acc), yerr = np.std(rate), fmt = "o", markersize = 8, c = color, label = label, capsize = 4)

            self.buffer_x.append(x)
            self.buffer_y.append(y)

        except OSError:
            print(f"Crunching numbers for {ensemble}: {dataset}...")
            NN = Ensemble(ensemble, supress_print = True)
            NN.money_plot(dataset)

            self.add(ensemble, dataset, **kwargs)

    def draw_line(self, **kwargs) -> None : 
        self.ax.plot(self.buffer_x, self.buffer_y, **kwargs)
        self.buffer_x, self.buffer_y = [], []

    def annotate(self, fcn : str, **kwargs) -> None :

        function = {
            "errorbar" : self.ax.errorbar,
            "scatter" : self.ax.scatter,
        }

        function[fcn]([], [], **kwargs)

    def __call__(self, xlim : tuple = (None, 1.05), ylim : tuple = (1, None)) -> None :
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        
        handles, labels = self.ax.get_legend_handles_labels()
        order = [i for i in range(len(handles))]
        order = [-1] + order[:-1]
        self.ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        plt.show()
