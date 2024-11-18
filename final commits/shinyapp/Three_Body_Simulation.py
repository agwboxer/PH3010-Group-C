import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class ThreeBodySimulation:
    def __init__(self, t_start=0, t_end=365, num_steps=100000):
        """
        Initializes the parameters for the three-body simulation

        Parameters:
        - t_start (floa): Start time of the simulation (in days | default = 0 days)
        - t_end (float): End time of the simulation (in days | default = 365 days)
        - num_steps (int): Number of time steps for the simulation (default = 100000)
        """
        # Constants
        self.G = 39.47841760435743  # Gravitational constant in AU^3 / (M_sun * day^2)
        self.M_sun = 1.0  # Mass of the Sun in Solar masses
        self.M_mercury = 1.652e-7  # Mass of Mercury in Solar masses
        self.M_venus = 2.447e-6    # Mass of Venus in Solar masses

        self.t_start = t_start
        self.t_end = t_end
        self.num_steps = num_steps

        # Orbital radii in AU
        self.r_mercury = 0.387
        self.r_venus = 0.723

        # Initial tangential velocities (circular approximation)
        self.v_mercury = np.sqrt(self.G * self.M_sun / self.r_mercury)  # in AU/day
        self.v_venus = np.sqrt(self.G * self.M_sun / self.r_venus)  # in AU/day

        # Initial conditions (AU, AU/day)
        self.initial_conditions = [
            0, 0, 0, 0, 0, 0,                # Sun: stationary at the origin
            self.r_mercury, 0, 0, 0, self.v_mercury, 0,  # Mercury: positioned along X-axis, moving along Y
            self.r_venus, 0, 0, 0, self.v_venus, 0      # Venus: positioned along X-axis, moving along Y
        ]

    def gravitational_acceleration(self, mass, pos1, pos2):
        """
        Calculate the gravitational acceleration on a body due to another body

        Parameters:
        - mass (float): Mass of the second body
        - pos1 (numpy array): Position of the first body
        - pos2 (numpy array): Position of the second body

        Returns:
        - numpy array: Gravitational acceleration experienced by the first body
        """
        r_vec = pos2 - pos1  # Vector from pos1 to pos2
        r_mag = np.linalg.norm(r_vec)  # Magnitude of the distance between the bodies
        if r_mag == 0:
            return np.zeros(3)  # Prevent division by zero if the positions are the same
        return self.G * mass * r_vec / r_mag**3  # Gravitational force divided by mass gives acceleration

    def three_body_dynamics(self, t, state):
        """
        Compute the derivatives (velocities and accelerations) for the three-body system.

        Parameters:
        - t (float): Time in the simulation
        - state (numpy aray): State vector containing positions and velocities of the bodies

        Returns:
        - derivatives (numpy array): Derivatives of the system's state (velocity and acceleration)
        """
        # Extract positions and velocities for each body from the state vector
        sun_pos = state[0:3]
        mercury_pos = state[6:9]
        venus_pos = state[12:15]
        
        sun_vel = state[3:6]
        mercury_vel = state[9:12]
        venus_vel = state[15:18]

        # Calculate accelerations due to gravitational forces between the bodies
        mercury_acc = self.gravitational_acceleration(self.M_sun, mercury_pos, sun_pos) + \
                      self.gravitational_acceleration(self.M_venus, mercury_pos, venus_pos)
        
        venus_acc = self.gravitational_acceleration(self.M_sun, venus_pos, sun_pos) + \
                    self.gravitational_acceleration(self.M_mercury, venus_pos, mercury_pos)

        # Sun remains stationary (approximation)
        sun_acc = np.zeros(3)
        
        # Concatenate the velocities and accelerations to return as derivatives for ODE solver
        derivatives = np.concatenate((sun_vel, sun_acc, mercury_vel, mercury_acc, venus_vel, venus_acc))
        return derivatives

    def run_simulation(self):
        """
        Run the simulationn and solve the system using an ODE solver

        Returns:
        - scipy.integrate.OdeResult: Solution to the differential equations
        """
        # Solve the system of differential equations for the three-body problem
        sol = solve_ivp(self.three_body_dynamics, (self.t_start, self.t_end), self.initial_conditions, 
                        t_eval=np.linspace(self.t_start, self.t_end, self.num_steps), 
                        rtol=1e-6, atol=1e-6, method='LSODA')
        return sol
