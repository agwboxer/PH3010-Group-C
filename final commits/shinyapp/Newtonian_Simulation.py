import numpy as np

class Newtonian_Orbit:
    """
    Simulates the orbit of a planet (e.g., Mercury) around the Sun using three numerical methods:
    Euler, 2nd-order Runge-Kutta (RK2), and 4th-order Runge-Kutta (RK4)

    Attributes:
    - ecc (float): Eccentricity of the orbit (default is Mercury's eccentricity 0.2056)
    - a (float): Semi-major axis of the orbit in astronomical units (default is Mercury's semi-major axis 0.387 AU)
    - steps (int): Number of time steps for the numerical simulation (default: 100)
    """

    def __init__(self, ecc=0.2056, a=0.387, steps=100):
        """
        Initialize the orbit simulation with given orbital parameters

        Parameters:
        - ecc (float): Eccentricity of the orbit
        - a (float): Semi-major axis in AU
        - steps (int): Number of simulation time steps
        """
        # Set orbital parameters
        self.ecc = ecc
        self.a = a
        self.b = self.a * np.sqrt(1 - self.ecc**2)  # Calculate the semi-minor axis
        self.steps = steps  # Number of time steps

        # Constants: Gravitational constant (in AU^3/yr^2/M_sun) and Sun's mass
        self.G = 4 * np.pi**2
        self.Ms = 1.0  # Mass of the Sun in solar masses

        # Calculate the orbital period (Kepler's 3rd law: T^2 = a^3 for units in AU and years)
        self.T = np.sqrt(self.a**3)
        
        # Define time step size
        self.dt = self.T / self.steps

        # Calculate perihelion (closest approach) and aphelion (farthest point) distances
        self.peri_d = self.a - self.ecc * self.a
        self.aphe_d = self.a + self.ecc * self.a

        # Calculate initial tangential velocity at perihelion using the vis-viva equation
        self.viy_calc = np.sqrt(
            ((2 * self.G) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d) ** 2)
        )

        # Set initial conditions: position and velocity at perihelion
        self.xi = -(self.a - self.ecc * self.a)  # Initial x-position
        self.yi = 0.0  # Initial y-position
        self.vix = 0  # Initial x-velocity
        self.viy = self.viy_calc  # Initial y-velocity

        # Store the initial conditions in an array
        self.initial_conditions = [self.xi, self.yi, self.vix, self.viy]

    def system_of_odes(self, state, t):
        """
        Defines the system of ordinary differential equations (ODEs) for the gravitational force

        Parameters:
        - state (array): Current position and velocity of the planet
        - t (float): Current time (not explicitly used here, but required for compatibility)

        Returns:
        - array: Velocity and acceleration in derivative notation [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)  # Calculate the distance from the Sun to the planet

        # Compute the derivatives based on Newton's law of gravitation
        dxdt = vx # Velocity in the x-direction
        dydt = vy
        dvxdt = -self.G * self.Ms * x / r**3  # Acceleration in the x-direction
        dvydt = -self.G * self.Ms * y / r**3  

        return np.array([dxdt, dydt, dvxdt, dvydt])

    def rk4(self):
        """
        Solves the ODEs using the 4th-order Runge-Kutta (RK4) method

        Returns:
        - tuple: Time array and solution array (x, y, vx, vy)
        """
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions  # Set initial state

        # Perform RK4 integration
        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)
            k3 = self.dt * self.system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * self.dt)
            k4 = self.dt * self.system_of_odes(x[i] + k3, t[i] + self.dt)
            x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, x

    def rk2(self):
        """
        Solves the ODEs using the 2nd-order Runge-Kutta (RK2) method

        Returns:
        - tuple: Time array and solution array (x, y, vx, vy)
        """
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions  

        # Perform RK2 integration
        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            k2 = self.dt * self.system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * self.dt)
            x[i + 1] = x[i] + k2

        return t, x

    def euler(self):
        """
        Solves the ODEs using the Euler method

        Returns:
        - tuple: Time array and solution array (x, y, vx, vy)
        """
        num_steps = int(self.T / self.dt)
        t = np.linspace(0, self.T, num_steps + 1)
        x = np.zeros((num_steps + 1, 4))
        x[0] = self.initial_conditions  

        # Perform Euler integration
        for i in range(num_steps):
            k1 = self.dt * self.system_of_odes(x[i], t[i])
            x[i + 1] = x[i] + k1

        return t, x
