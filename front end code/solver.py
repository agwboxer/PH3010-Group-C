import numpy as np

class Simulator:
    def __init__(self, Ecc=0.2056, a=0.387, Ms=1.0):
        # Initialize system parameters
        self.Ecc = Ecc                # Eccentricity of orbit (Mercury)
        self.a = a                    # Semi-major axis
        self.Ms = Ms                  # Mass of the Sun in solar mass units
        self.G = 4 * np.pi**2         # Gravitational constant G (AU^3 / yr^2 / solar mass)
        
        # Calculate derived parameters
        self.b = self.a * np.sqrt(1 - self.Ecc**2)     # Semi-minor axis
        self.Tsq = self.a**3                           # Orbit Period squared
        self.T = np.sqrt(self.Tsq)                     # Orbit period

        # Calculate perihelion and aphelion distances
        self.peri_d = self.a - self.Ecc * self.a
        self.aphe_d = self.a + self.Ecc * self.a

        # Calculate initial velocity at perihelion
        self.viy_calc = np.sqrt(((2 * 4 * np.pi**2) * (1 / self.peri_d - 1 / self.aphe_d)) /
                                (1 - (self.peri_d / self.aphe_d)**2))
        
        # Initialize initial conditions
        self.initial_conditions = [-self.peri_d, 0.0, 0.0, self.viy_calc]

    def set_initial_conditions(self, xi, yi, vix, viy):
        """Update initial conditions."""
        self.initial_conditions = [xi, yi, vix, viy]

    def radius(self, x, y):
        """Calculate the current radius."""
        return np.sqrt(x**2 + y**2)

    def system_of_odes(self, state, t):
        """Defines the system of ODEs for the two-body problem."""
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r**3
        dvydt = -self.G * self.Ms * y / r**3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def Euler(self, t0, t_final, dt):
        """Euler method for solving the system."""
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x

    def RK2(self, t0, t_final, dt):
        """Second-order Runge-Kutta method."""
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            x[i + 1] = x[i] + k2

        return t, x

    def RK4(self, t0, t_final, dt):
        """Fourth-order Runge-Kutta method."""
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



simulation = TwoBodySimulation()


simulation.set_initial_conditions(-0.3, 0.0, 0.0, 8.0)


dt = simulation.T / 100
t0, t_final = 0, simulation.T


t_euler, euler_sol = simulation.Euler(t0, t_final, dt)
t_rk2, rk2_sol = simulation.RK2(t0, t_final, dt)
t_rk4, rk4_sol = simulation.RK4(t0, t_final, dt)


x_euler, y_euler = euler_sol[:, 0], euler_sol[:, 1]
x_rk2, y_rk2 = rk2_sol[:, 0], rk2_sol[:, 1]
x_rk4, y_rk4 = rk4_sol[:, 0], rk4_sol[:, 1]

