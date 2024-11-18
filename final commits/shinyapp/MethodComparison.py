import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Numerical_Comparison:
    """
    Class to simulate and compare numerical integration methods for solving 
    the equations of motion of Mercury under the gravitational influence of the Sun.

    Parameters:
    - ecc (float): Eccentricity of Mercury's orbit (default: 0.2056)
    - a (float): Semi-major axis of Mercury's orbit in astronomical units (default: 0.387)
    """
    
    def __init__(self, ecc=0.2056, a=0.387):
        # Set the orbital parameters
        self.Ecc = ecc  # Eccentricity of orbit
        self.a = a  # Semi-major axis (AU)
        self.b = self.a * np.sqrt(1 - self.Ecc**2)  # Semi-minor axis (AU)
        self.Tsq = self.a**3  # Orbital period squared (yr^2)
        self.T = np.sqrt(self.Tsq)  # Orbital period (yr)

        # Calculate perihelion and aphelion distances
        self.peri_d = self.a - self.Ecc * self.a
        self.aphe_d = self.a + self.Ecc * self.a

        # Calculate initial velocity at perihelion using the vis-viva equation
        self.viy_calc = np.sqrt(
            ((2 * 4 * np.pi**2) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d)**2)
        )

        # Initial conditions: position and velocity at perihelion
        self.xi = -(self.a - self.Ecc * self.a)
        self.yi = 0.0
        self.vix = 0
        self.viy = self.viy_calc

        # Constants
        self.Ms = 1.0  # Mass of the sun (M_sun)
        self.G = 4 * np.pi**2  # Gravitational constant (AU^3/yr^2/M_sun)

        # Simulation parameters
        self.steps = 5000  # Number of steps in the simulation
        self.dt = self.T / self.steps  # Step size
        self.t = np.arange(0.0, self.T, self.dt)  # Time array

    def radius(self, x, y):
        """
        Calculate the radius from the Sun to the current position.

        Parameters:
        - x (float): x-coordinate
        - y (float): y-coordinate

        Returns:
        - float: Distance from the origin (Sun)
        """
        return np.sqrt(x ** 2 + y ** 2)

    def system_of_odes(self, state, t):
        """
        Define the system of ordinary differential equations for the motion of Mercury.

        Parameters:
        - state (np.array): Array [x, y, vx, vy]
        - t (float): Current time (not used explicitly)

        Returns:
        - np.array: Derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state
        r = self.radius(x, y)
        dxdt = vx
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r ** 3
        dvydt = -self.G * self.Ms * y / r ** 3
        return np.array([dxdt, dydt, dvxdt, dvydt])

    def RK2(self, initial_conditions, t0, t_final, h):
        """
        Perform integration using the 2nd-order Runge-Kutta method.

        Parameters:
        - initial_conditions (list): Initial state [x, y, vx, vy]
        - t0 (float): Start time
        - t_final (float): End time
        - h (float): Time step size

        Returns:
        - tuple: Time array and solution array
        """
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
        """
        Perform integration using the Euler method.

        Parameters:
        - initial_conditions (list): Initial state [x, y, vx, vy]
        - t0 (float): Start time
        - t_final (float): End time
        - dt (float): Time step size

        Returns:
        - tuple: Time array and solution array
        """
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x

    def RK4(self, initial_conditions, t0, t_final, dt):
        """
        Perform integration using the 4th-order Runge-Kutta method.

        Parameters:
        - initial_conditions (list): Initial state [x, y, vx, vy]
        - t0 (float): Start time
        - t_final (float): End time
        - dt (float): Time step size

        Returns:
        - tuple: Time array and solution array
        """
        num_steps = int((t_final - t0) / dt)
        t = np.linspace(t0, t_final, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = initial_conditions

        for i in range(num_steps):
            k1 = dt * self.system_of_odes(x[i], t[i])
            k2 = dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
            k3 = dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
            k4 = dt * self.system_of_odes(x[i] + k3, t[i] + dt)
            x[i + 1] = x[i] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        return t, x
