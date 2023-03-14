from .__config__ import *
from .Signal import *
from .Generator import *
from .Classifier import *

# plot the estimated confidence range of provided classifiers
def confidence_comparison(confidence_level, *args, **kwargs):

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