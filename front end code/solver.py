import numpy as np

class PlanetOrbit:
    """
    A class that simulates the orbit of a planet around a star using numerical methods.

    Parameters:
    - ecc (float): Eccentricity of the orbit (default 0.2056).
    - a (float): Semi-major axis of the orbit in Astronomical Units (AU) (default 0.387).
    - steps (int): Number of time steps in the simulation (default 100).
    """
    
    def __init__(self, ecc=0.2056, a=0.387, steps=100):
        """
        Initializes the parameters of the orbit and sets up initial conditions.

        Parameters:
        - ecc (float): Eccentricity of the orbit (default 0.2056).
        - a (float): Semi-major axis of the orbit in AU (default 0.387).
        - steps (int): Number of time steps for the simulation (default 100).
        """
        self.ecc = ecc  # Eccentricity of the orbit
        self.a = a  # Semi-major axis (AU)
        self.b = self.a * np.sqrt(1 - self.ecc**2)  # Semi-minor axis
        self.steps = steps  # Number of time steps
        self.G = 4 * np.pi**2  # Gravitational constant (AU^3/year^2)
        self.Ms = 1.0  # Mass of the Sun (solar masses)
        self.T = np.sqrt(self.a**3)  # Orbital period (in years)
        self.dt = self.T / self.steps  # Time step
        self.peri_d = self.a - self.ecc * self.a  # Perihelion distance
        self.aphe_d = self.a + self.ecc * self.a  # Aphelion distance
        self.viy_calc = np.sqrt(((2 * self.G) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d) ** 2))  # Initial velocity calculation
        self.xi = -(self.a - self.ecc * self.a)  # Initial x-position (perihelion)
        self.yi = 0.0  # Initial y-position
        self.vix = 0  # Initial x-velocity
        self.viy = self.viy_calc  # Initial y-velocity
        self.initial_conditions = [self.xi, self.yi, self.vix, self.viy]  # Initial conditions for the simulation

    def system_of_odes(self, state, t):
        """
        Calculate the derivatives of position and velocity for the planetary orbit.

        Parameters:
        - state (array): An array containing the current position (x, y) and velocity (vx, vy).
        - t (float): The current time (used for integration, but not needed for the ODEs themselves).

        Returns:
        - np.array: The derivatives [dxdt, dydt, dvxdt, dvydt], representing the rates of change for position and velocity.
        """
        x, y, vx, vy = state  # Extract position and velocity
        r = np.sqrt(x**2 + y**2)  # Distance from the center (star)
        dxdt = vx  # Derivative of x (velocity in x-direction)
        dydt = vy  # Derivative of y (velocity in y-direction)
        dvxdt = -self.G * self.Ms * x / r**3  # Derivative of velocity in x (gravitational force)
        dvydt = -self.G * self.Ms * y / r**3  # Derivative of velocity in y (gravitational force)
        return np.array([dxdt, dydt, dvxdt, dvydt])  # Return the system of ODEs

    def rk4(self):
        """
        Solve the orbital system using the 4th-order Runge-Kutta (RK4) method.

        Returns:
        - tuple: A tuple containing:
            - t (array): An array of time points.
            - x (array): A 2D array with positions and velocities at each time step.
        """
        num_steps = int(self.T / self.dt)  # Number of time steps
        t = np.linspace(0, self.T, num_steps + 1)  # Time array from 0 to T
        x = np.zeros((num_steps + 1, 4))  # Array to store position and velocity at each time step
        x[0] = self.initial_conditions  # Set initial conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])  # Calculate k1
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)  # Calculate k2
            k3 = self.dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * self.dt)  # Calculate k3
            k4 = self.dt * self.system_of_odes(x[i] + k3, t[i] + self.dt)  # Calculate k4
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6  # Update position and velocity

        return t, x  # Return time and solution arrays

    def rk2(self):
        """
        Solve the orbital system using the 2nd-order Runge-Kutta (RK2) method.

        Returns:
        - tuple: A tuple containing:
            - t (array): An array of time points.
            - x (array): A 2D array with positions and velocities at each time step.
        """
        num_steps = int(self.T / self.dt)  # Number of time steps
        t = np.linspace(0, self.T, num_steps + 1)  # Time array from 0 to T
        x = np.zeros((num_steps + 1, 4))  # Array to store position and velocity at each time step
        x[0] = self.initial_conditions  # Set initial conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])  # Calculate k1
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)  # Calculate k2
            x[i + 1] = x[i] + k2  # Update position and velocity

        return t, x  # Return time and solution arrays

    def euler(self):
        """
        Solve the orbital system using the Euler method, a first-order numerical method.

        Returns:
        - tuple: A tuple containing:
            - t (array): An array of time points.
            - x (array): A 2D array with positions and velocities at each time step.
        """
        num_steps = int(self.T / self.dt)  # Number of time steps
        t = np.linspace(0, self.T, num_steps + 1)  # Time array from 0 to T
        x = np.zeros((num_steps + 1, 4))  # Array to store position and velocity at each time step
        x[0] = self.initial_conditions  # Set initial conditions

        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])  # Calculate k1
            x[i + 1] = x[i] + k1  # Update position and velocity using Euler method

        return t, x  # Return time and solution arrays
