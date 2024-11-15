import numpy as np

class MercuryOrbit:
    def __init__(self, ecc=0.2056, a=0.387, steps=100):
        self.ecc = ecc
        self.a = a
        self.b = self.a * np.sqrt(1 - self.ecc**2)
        self.steps = steps
        self.G = 4 * np.pi**2
        self.Ms = 1.0
        self.T = np.sqrt(self.a**3)
        self.dt = self.T / self.steps
        self.peri_d = self.a - self.ecc * self.a
        self.aphe_d = self.a + self.ecc * self.a
        self.viy_calc = np.sqrt(((2 * self.G) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d) ** 2))
        self.xi = -(self.a - self.ecc * self.a)
        self.yi = 0.0
        self.vix = 0
        self.viy = self.viy_calc
        self.initial_conditions = [self.xi, self.yi, self.vix, self.viy]

    def system_of_odes(self, state, t):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r**3
        dvydt = -self.G * self.Ms * y / r**3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def rk4(self):
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)
            k3 = self.dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * self.dt)
            k4 = self.dt * self.system_of_odes(x[i] + k3, t[i] + self.dt)
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, x

    def rk2(self):
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)
            x[i + 1] = x[i] + k2

        return t, x

    def euler(self):
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x
