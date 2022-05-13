import matplotlib.pyplot as plt
import numpy as np
from showers import Line

binary_cmap = plt.cm.get_cmap("binary", 100)

class Layer():

    def __init__(self, x : int, size : int, separation : float):

        self.x = x
        self.ys = np.linspace(-size/2 * separation, size/2 * separation, size)

        for y in self.ys:

            color = binary_cmap(np.random.randint(30, 100))
            ax.add_patch(plt.Circle((x,y), 0.5, fc = color, ec = "k"))

    def connect(self, other):

        x_i = other.x
        x_f = self.x

        for y_f in self.ys:
            for y_i in other.ys:
                plt.plot([x_i, x_f],[y_i, y_f], c = "#bf4040", lw = 1, zorder=0)

    def inject(self):

        for y_f in self.ys:

            n_injected = np.random.randint(1,10)
            for n in range(n_injected):
                length = np.random.uniform(7,15)
                theta = np.random.normal(np.pi + 0.2 * (length * y_f) * np.pi/180, 15 * np.pi/180)

                line = Line((self.x,y_f), theta, length)
                line()


class Network():

    def __init__(self, architecture : tuple):

        layers = []

        for i, size in enumerate(architecture):

            layer = Layer(i * 3, size, 0.9)
            layers.append(layer)

            if i != 0:
                layer.connect(layers[i-1])
            # else:
            #     layer.inject()

fig, ax = plt.subplots()
Network((9,2))
ax.set_aspect("equal")
plt.axis("off")
plt.show()

