import numpy as np
import matplotlib.pyplot as plt

class OrbitalSimulation:
    def __init__(self):
        # Initial conditions, all initial values taken at the perihelion in standard astronomical units
        self.Ecc = 0.2056  # Eccentricity of orbit (Mercury)
        self.a = 0.387  # Semi-major axis distance (AU)
        self.b = self.a * np.sqrt(1 - self.Ecc**2)  # Semi-minor axis (AU)
        self.Tsq = self.a**3  # Orbit Period squared (yr^2)
        self.T = np.sqrt(self.Tsq)  # Orbit period (yr)

        self.peri_d = self.a - self.Ecc * self.a
        self.aphe_d = self.a + self.Ecc * self.a

        self.viy_calc = np.sqrt(
            ((2 * 4 * np.pi**2) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d)**2)
        )

        # Initial positions and velocities
        self.xi = -(self.a - self.Ecc * self.a)
        self.yi = 0.0
        self.vix = 0
        self.viy = self.viy_calc

        self.Ms = 1.0  # Mass of the sun in solar mass units
        self.G = 4 * np.pi**2  # Gravitational constant G (AU^3/yr^2/M_sun)

        self.initial_conditions = [self.xi, self.yi, self.vix, self.viy]

    def radius(self, x, y):
        """Calculate the current radius of the star."""
        return np.sqrt(x**2 + y**2)

    def system_of_odes(self, state, t):
        """Define the system of ODEs for the orbit."""
        x, y, vx, vy = state
        r = self.radius(x, y)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r**3
        dvydt = -self.G * self.Ms * y / r**3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def euler(self, t0, t_final, dt):
        """Solve using the Euler method."""
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x

    def rk2(self, t0, t_final, dt):
        """Solve using the RK2 method."""
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            x[i + 1] = x[i] + k2

        return t, x

    def rk4(self, t0, t_final, dt):
        """Solve using the RK4 method."""
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            k3 = dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
            k4 = dt * self.system_of_odes(x[i] + k3, t[i] + dt)
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, x

    def run_simulation(self, steps_data):
        """Run the simulation for different time steps and plot the results."""
        store_E_dist = []
        store_RK2_dist = []
        store_RK4_dist = []
        store_t_data = []

        for val in steps_data:
            dt = self.T / val

            _, EulerSoln = self.euler(0, self.T, dt)
            distance_E = self.radius(-self.a - EulerSoln[-1, 0], EulerSoln[-1, 1])
            store_E_dist.append(distance_E)

            _, RK2Soln = self.rk2(0, self.T, dt)
            distance_RK2 = self.radius(-self.a - RK2Soln[-1, 0], RK2Soln[-1, 1])
            store_RK2_dist.append(distance_RK2)

            _, RK4Soln = self.rk4(0, self.T, dt)
            distance_RK4 = self.radius(-self.a - RK4Soln[-1, 0], RK4Soln[-1, 1])
            store_RK4_dist.append(distance_RK4)

            store_t_data.append(val)

        self.plot_results(store_t_data, store_E_dist, store_RK2_dist, store_RK4_dist)

    def plot_results(self, store_t_data, store_E_dist, store_RK2_dist, store_RK4_dist):
        """Plot the results."""
        plt.figure()
        plt.scatter(store_t_data, store_E_dist, label='Euler', marker='.')
        plt.scatter(store_t_data, store_RK2_dist, label='RK2', marker='.')
        plt.scatter(store_t_data, store_RK4_dist, label='RK4', marker='.')
        plt.xlabel('No. of timesteps')
        plt.ylabel('Distance from start (AU)')
        plt.title('Effect of timestep on orbital accuracy')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    simulation = OrbitalSimulation()
    steps_data = np.linspace(10, 500, 50)
    simulation.run_simulation(steps_data)
