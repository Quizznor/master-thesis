from scipy.stats import logistic
import matplotlib.pyplot as plt
import numpy as np

E_critical = 0.1
dTheta = 10 * np.pi/180
Theta_initial = -50 * np.pi/180

class Line():

    def __init__(self, start : tuple, theta : float, length : float):

        self.theta = theta
        self.x_i, self.y_i = start
        self.x_f = self.x_i + length * np.cos(theta)
        self.y_f = self.y_i + length * np.sin(theta)

    def __call__(self):
        if self.y_f >= 0:
            plt.plot([self.x_i, self.x_f], [self.y_i, self.y_f], c = "#bf4040", lw = 0.5)
        else:
            plt.plot([self.x_i, self.x_i + self.y_i * np.tan(np.pi/2 + self.theta)], [self.y_i, np.random.normal(0,0.1)], c = "#bf4040", lw = 0.5)

    def get_endpoint(self):
        return (self.x_f, self.y_f)


class Shower():

    def __init__(self, E_0 : float, position : tuple, Theta : int, thinning = 0.3):

        if position[1] > 0 and E_0 > E_critical:

            particle = Line(position, Theta, np.random.uniform(0, 3 - 0.8 * position[1]))
            endpoint = particle.get_endpoint()
            particle()

            # 2 subsequent branches
            n_branches = 5
            for i in range(n_branches):

                 # random chance that part of the shower dies
                if np.random.uniform(0, 1) > thinning:
                    angle = np.random.normal(Theta, dTheta)
                    Theta = angle if angle <= Theta_initial + 30 * np.pi/180 else Theta_initial
                    # Theta = angle if angle >= -90 * np.pi/180 else Theta_initial
                    Shower(E_0 / n_branches, endpoint, Theta)

if __name__ == "__main__":
    test = Shower(30000,(0,3), Theta_initial)
    plt.axis('off')
    plt.show()