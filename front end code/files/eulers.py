import matplotlib.pyplot as plt
import numpy as np

class MercuryOrbitSimulator:
    def __init__(self, xi=-0.307278, yi=0.0, vix=0, viy=12.0, Ms=1.0, a=0.387, steps=25000):

        # Initial conditions
        self.xi = xi  # initial x position
        self.yi = yi  # initial y position
        self.vix = vix  # initial x velocity
        self.viy = viy  # initial y velocity
        self.Ms = Ms  # mass of the sun in solar mass units
        self.G = 4 * np.pi ** 2  # gravitational constant
        self.a = a  # semi-major distance (AU)

        # Derived values
        self.Tsq = self.a ** 3  # orbital period squared
        self.T = np.sqrt(self.Tsq)  # orbital period
        self.steps = steps  # number of steps in simulation
        self.dt = self.T / self.steps  # time increment
        self.t = np.arange(0.0, self.T, self.dt)  # time array

        # Position of the sun
        self.xs = 0.0
        self.ys = 0.0

        # Lists for storing simulation results
        self.xpos_list = []
        self.ypos_list = []
        self.vx_list = []
        self.vy_list = []

        # Initial values for position and velocity
        self.xpos = self.xi
        self.ypos = self.yi
        self.vx = self.vix
        self.vy = self.viy

    def radius(self, x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def SecondDev(self, position, xpos, ypos):
        return -self.G * self.Ms * (position - self.xs) / (self.radius(xpos, ypos)) ** 3

    def EulerFirst(self, x0, v0):
        return x0 + self.dt * v0

    def EulerSecond(self, v0, f):
        return v0 + self.dt * f

    def simulate(self):
        # Initialize the base variables for the loop
        base_vars = self.xpos, self.ypos, self.vx, self.vy

        # Loop over each time step to calculate position and velocity
        for i, val in enumerate(self.t):
            # x-axis calculations
            vx = self.EulerSecond(base_vars[2], self.SecondDev(base_vars[0], base_vars[0], base_vars[1]))
            xpos = self.EulerFirst(base_vars[0], base_vars[2])
            self.vx_list.append(vx)
            self.xpos_list.append(xpos)

            # y-axis calculations
            vy = self.EulerSecond(base_vars[3], self.SecondDev(base_vars[1], base_vars[0], base_vars[1]))
            ypos = self.EulerFirst(base_vars[1], base_vars[3])
            self.vy_list.append(vy)
            self.ypos_list.append(ypos)

            # Update base variables for the next iteration
            base_vars = xpos, ypos, vx, vy

    def plot_orbit(self):
        # Plot the orbit of Mercury and the Sun
        plt.figure()
        plt.plot(self.xpos_list, self.ypos_list, label="Mercury's Orbit")
        plt.scatter(self.xs, self.ys, marker='o', label='Sun', color='orange')
        plt.legend(loc='lower right', fontsize=8, frameon=False)
        plt.xlabel("x position (AU)")
        plt.ylabel("y position (AU)")
        plt.title("Orbit of Mercury around the Sun")
        plt.show()

    def plot_position(self):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Plotting x-position over time in the first subplot
        ax[0].plot(self.t, self.xpos_list, color='orange', label="x position")
        ax[0].set_xlabel("Time (years)")
        ax[0].set_ylabel("Position (AU)")
        ax[0].legend(loc="lower right", fontsize=8, frameon=False)

        # Plotting y-position over time in the second subplot
        ax[1].plot(self.t, self.ypos_list, color='green', label="y position")
        ax[1].set_xlabel("Time (years)")
        ax[1].set_ylabel("Position (AU)")
        ax[1].legend(loc="lower right", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.show()

    def plot_velocity(self):
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))

        # Plotting x-velocity over time
        ax[0].plot(self.t, self.vx_list, color='blue', label="x velocity")
        ax[0].set_xlabel("Time (years)")
        ax[0].set_ylabel("x Velocity (AU/year)")
        ax[0].legend(loc="lower right", fontsize=8, frameon=False)

        # Plotting y-velocity over time
        ax[1].plot(self.t, self.vy_list, color='red', label="y velocity")
        ax[1].set_xlabel("Time (years)")
        ax[1].set_ylabel("y Velocity (AU/year)")
        ax[1].legend(loc="lower right", fontsize=8, frameon=False)

        plt.tight_layout()
        plt.show()

# Instantiate and run the simulator with optional custom initial conditions
simulator = MercuryOrbitSimulator(xi=-0.307278, yi=0.0, vix=0, viy=12.0, Ms=1.0, a=0.387, steps=25000)
simulator.simulate()
simulator.plot_orbit()
simulator.plot_position()
simulator.plot_velocity()
