import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Numerical_Comparison:
    def __init__(self, ecc=0.2056, a=0.387):
        # Initial conditions, all initial values taken at perihelion in standard astronomical units
        self.Ecc = ecc  # Eccentricity of orbit (Mercury)
        self.a = a  # Semi-major axis distance (AU)
        self.b = self.a * np.sqrt(1 - self.Ecc**2)
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

        self.steps = 5000  # Number of steps in the simulation
        self.dt = self.T / self.steps  # Length of a step
        self.t = np.arange(0.0, self.T, self.dt)  # Time array

    def radius(self, x, y):
        """
        Returns the radius from the origin (Sun) to the current position of Mercury.
        """
        return np.sqrt(x ** 2 + y ** 2)

    def system_of_odes(self, state, t):
        """
        Defines the system of ordinary differential equations for the motion of Mercury
        under gravitational influence of the Sun.
        """
        x, y, vx, vy = state
        r = np.sqrt(x ** 2 + y ** 2)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r ** 3
        dvydt = -self.G * self.Ms * y / r ** 3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def RK2(self, initial_conditions, t0, t_final, h):
        num_steps = int((t_final - t0) / h)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))  
        x[0] = initial_conditions

        for i in range(num_steps):
            k1 = h * self.system_of_odes(x[i], t[i])
            k2 = h * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * h)
            x[i + 1] = x[i] + k2

        return t, x

    def Euler(self, initial_conditions, t0, t_final, dt):
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))  
        x[0] = initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x

    def RK4(self, initial_conditions, t0, t_final, dt):
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))  
        x[0] = initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            k3 = dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
            k4 = dt * self.system_of_odes(x[i] + k3, t[i] + 0.5 * dt)
            x[i + 1] = x[i] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        return t, x

    def run_simulation(self, steps_data):
        store_E_dist = []
        store_RK2_dist = []
        store_RK4_dist = []
        store_t_data = []

        for i, val in enumerate(steps_data):
            dt = self.T / steps_data[i]

            Eulert, EulerSoln = self.Euler([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            final_EData = EulerSoln[-1]
            distance_E = np.sqrt((-self.a - final_EData[0]) ** 2 + final_EData[1] ** 2)
            store_E_dist.append(distance_E)

            RK2t, RK2Soln = self.RK2([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            final_RK2Data = RK2Soln[-1]
            distance_RK2 = np.sqrt((-self.a - final_RK2Data[0]) ** 2 + final_RK2Data[1] ** 2)
            store_RK2_dist.append(distance_RK2)

            RK4t, RK4Soln = self.RK4([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            final_RK4Data = RK4Soln[-1]
            distance_RK4 = np.sqrt((-self.a - final_RK4Data[0]) ** 2 + final_RK4Data[1] ** 2)
            store_RK4_dist.append(distance_RK4)

            store_t_data.append(steps_data[i])

        return store_t_data, store_E_dist, store_RK2_dist, store_RK4_dist



    def compute_errors(self, steps_data):
        steps = 100000
        dt = self.T / steps
        _, solution = self.RK4([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
        x, y = solution[-1, 0], solution[-1, 1]
        pos = np.sqrt(x ** 2 + y ** 2)

        global_error_euler = []
        global_error_rk2 = []
        global_error_rk4 = []
        step_sizes = []

        for val in steps_data:
            dt = self.T / val
            step_sizes.append(dt)

            _, euler_sol = self.Euler([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            euler_x, euler_y = euler_sol[-1, 0], euler_sol[-1, 1]
            euler_pos = np.sqrt(euler_x ** 2 + euler_y ** 2)
            global_error_euler.append(abs(euler_pos - pos))

            _, rk2_sol = self.RK2([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            rk2_x, rk2_y = rk2_sol[-1, 0], rk2_sol[-1, 1]
            rk2_pos = np.sqrt(rk2_x ** 2 + rk2_y ** 2)
            global_error_rk2.append(abs(rk2_pos - pos))

            _, rk4_sol = self.RK4([self.xi, self.yi, self.vix, self.viy], 0, self.T, dt)
            rk4_x, rk4_y = rk4_sol[-1, 0], rk4_sol[-1, 1]
            rk4_pos = np.sqrt(rk4_x ** 2 + rk4_y ** 2)
            global_error_rk4.append(abs(rk4_pos - pos))

        return global_error_euler, global_error_rk2, global_error_rk4, step_sizes

