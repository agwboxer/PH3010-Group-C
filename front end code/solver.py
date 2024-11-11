import numpy as np

class TwoBodySimulation:
    def __init__(self, Ecc=0.2056, a=0.387, Ms=1.0):
        self.Ecc = Ecc
        self.a = a
        self.Ms = Ms
        self.G = 4 * np.pi**2
        self.b = self.a * np.sqrt(1 - self.Ecc**2)
        self.Tsq = self.a**3
        self.T = np.sqrt(self.Tsq)
        self.peri_d = self.a - self.Ecc * self.a
        self.aphe_d = self.a + self.Ecc * self.a
        self.viy_calc = np.sqrt(((2 * 4 * np.pi**2) * (1 / self.peri_d - 1 / self.aphe_d)) /
                                (1 - (self.peri_d / self.aphe_d)**2))
        self.initial_conditions = [-self.peri_d, 0.0, 0.0, self.viy_calc]

    def set_initial_conditions(self, xi, yi, vix, viy):
        self.initial_conditions = [xi, yi, vix, viy]

    def system_of_odes(self, state, t):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r**3
        dvydt = -self.G * self.Ms * y / r**3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def RK4(self, t0, t_final, dt):
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions
        radii = []

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            k3 = dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
            k4 = dt * self.system_of_odes(x[i] + k3, t[i] + dt)
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            radii.append(np.sqrt(x[i + 1, 0]**2 + x[i + 1, 1]**2))

        return t, radii



if __name__ == "__main__":
    simulation = TwoBodySimulation()
    dt = simulation.T / 100
    t, radii = simulation.RK4(0, simulation.T, dt)

    np.save("radii_data.npy", radii)
