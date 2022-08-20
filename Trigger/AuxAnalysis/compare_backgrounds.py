import TriggerStudyBinaries_v2 as old
import TriggerStudyBinaries_v6 as new
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 22})

# TODO implement correlation?
class Histogram2d():

    def __new__(self, x : np.ndarray, y : np.ndarray, **kwargs) -> plt.Figure : 

        fig = plt.figure(figsize = (20,20))

        heatmap = plt.subplot2grid((7, 7), (2, 0), colspan=5, rowspan = 5)
        histo_x = plt.subplot2grid((7, 7), (0, 0), colspan=5, rowspan = 2, sharex = heatmap)
        histo_y = plt.subplot2grid((7, 7), (2, 5), rowspan=5, colspan = 2, sharey = heatmap)
        legend = plt.subplot2grid((7,7), (0,5), colspan = 2, rowspan = 2)
        legend.axis("off"), histo_x.axis("off"), histo_y.axis("off")
        
        x_unit = kwargs.get("labels", "")[0].split('/')[-1]
        y_unit = kwargs.get("labels", "")[1].split('/')[-1]
        heatmap.set_xlabel(kwargs.get("labels","x")[0])
        heatmap.set_ylabel(kwargs.get("labels","y")[1])
        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)
        correlation = np.corrcoef(x, y)
        n_bins = kwargs.get("bins", 40)
        x_scale, y_scale = kwargs.get("scale", ('linear', 'linear'))

        bin_maker = \
            {
                "linear" : np.linspace,
                "log" : np.geomspace
            }

        heatmap.set_xscale(x_scale), heatmap.set_yscale(y_scale)
        x_bins = bin_maker[x_scale](x_mean - 4 * x_std, x_mean + 4 * x_std, n_bins)
        y_bins = bin_maker[y_scale](y_mean - 4 * y_std, y_mean + 4 * y_std, n_bins) 

        information = [r"$\langle x \rangle$ = " + f"{x_mean:.2f}" + x_unit,
                       r" $\sigma_x$ = " + f"{x_std:.2f}" + x_unit,
                       r"$\langle y \rangle$ = " + f"{y_mean:.2f}" + y_unit,
                       r" $\sigma_y$ = " + f"{y_std:.2f}" + y_unit,
                       r"$\rho_{xy}$ = " + f"{correlation[0,1]:.3f}"
                       ]

        for i, y_value in enumerate(np.linspace(0.08, 0.73, 5)):
            legend.annotate(information[::-1][i], (0, y_value))

        heatmap.hist2d(x,y, bins = (x_bins, y_bins))
        histo_y.hist(y, bins = y_bins, histtype = "step", color = "steelblue", lw = 2, orientation = "horizontal")
        histo_x.hist(x, bins = x_bins, histtype = "step", color = "steelblue", lw = 2)

        heatmap.scatter(x_mean, y_mean, marker = "x", color = "white", s = 50)
        
        one_sigma = self.draw_ellipse(heatmap, (x_mean, y_mean), (x_std, y_std), 1)
        two_sigma = self.draw_ellipse(heatmap, (x_mean, y_mean), (x_std, y_std), 2)
        three_sigma = self.draw_ellipse(heatmap, (x_mean, y_mean), (x_std, y_std), 3)

        plt.subplots_adjust(wspace = 0.02, hspace = 0.02)
        
        save_fig = kwargs.get("savefig", "")
        if save_fig: plt.savefig(save_fig + ".png")

        return fig

    @staticmethod
    def draw_ellipse(axis : plt.Axes, loc : tuple, radii : tuple, interval : float) -> None : 

        angles = np.linspace(0,2*np.pi,100)
        X = interval * radii[0] * np.cos(angles) + loc[0]
        Y = interval * radii[1] * np.sin(angles) + loc[1]

        axis.plot(X, Y, ls = "--", color = "white", lw = 0.5)
        axis.annotate(f"{interval}" + r" $\sigma$", xy = (1.005 * X[15], 1.005 * Y[15]), color = "white", fontsize = 14)

TestRealBackground = old.EventGenerator("all", real_background = True, split = 1, prior = 0, ADC_to_VEM = 1, force_inject = 0)
TestModelBackground = old.EventGenerator("all", real_background = False, split = 1, prior = 0, ADC_to_VEM = 1, force_inject = 0)
RealBackgroundCorrected = new.EventGenerator("all", real_background = True, split = 1, prior = 0, adc_to_vem = 1, force_inject = 0)

l = ["Random traces, baseline corrected"]

for i, Dataset in enumerate([RealBackgroundCorrected]):

    Dataset.files = np.zeros(100000)
    histo_mean, histo_std = [], []

    for batch in range(Dataset.__len__()):

        print(f"Fetching batch {batch + 1}/{Dataset.__len__()}: {100 * (batch/Dataset.__len__()):.2f}%", end = "...\r")

        traces, _ = Dataset.__getitem__(batch)

        for trace in traces:

            # to not include signal, impose (4 sigma * 2 ADC std) = +-8 ADC cut
            trace = np.clip(trace, np.mean(trace) - 8, np.mean(trace) + 8)

            mean, std = np.mean(trace), np.std(trace)
            histo_mean.append(mean)
            histo_std.append(std)

    Histogram2d(np.array(histo_mean), np.array(histo_std), labels = ["Baseline mean / ADC", "Baseline std / ADC"], savefig = l[i])

    print(f"{l[i]}: mu = {np.mean(histo_mean):.3f}, std = {np.mean(histo_std):.3f}")

    plt.show()