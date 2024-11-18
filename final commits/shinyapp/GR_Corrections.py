import numpy as np
import matplotlib.pyplot as plt

class GR_Orbit:
    """
    Class to simulate the orbit of a body under the influence of gravity
    with an option to include General Relativity (GR) corrections.

    Parameters:
    - G (float): Gravitational constant (default: 39.478)
    - M (float): Mass of the central body (default: 1)
    - c_prime (float): Modified speed of light for GR effects (default: 63.239)
    - x0 (float): Initial x-coordinate (default: 0.307)
    - y0 (float): Initial y-coordinate (default: 0)
    - vx0 (float): Initial velocity in the x-direction (default: 0)
    - vy0 (float): Initial velocity in the y-direction (default: 12.4375)
    """
    def __init__(self, G=39.478, M=1, c_prime=63.239, x0=0.307, y0=0, vx0=0, vy0=12.4375):
        self.G = G
        self.M = M
        self.c_prime = c_prime  # Modified speed of light
        self.initial_state = np.array([x0, y0, vx0, vy0])

    def derivatives(self, t, state, include_gr=True):
        """
        Calculate the derivatives for the state [x, y, vx, vy].

        Parameters:
        - t (float): Time (not explicitly used here but included for generality)
        - state (np.array): Current state of the system [x, y, vx, vy]
        - include_gr (bool): Whether to include General Relativity corrections

        Returns:
        - np.array: Array containing derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)  # Distance from the central mass

        # Calculate Newtonian accelerations
        ax = -self.G * self.M * x / r**3
        ay = -self.G * self.M * y / r**3

        # Apply General Relativity correction to accelerations, if enabled
        if include_gr:
            gr_correction = -3 * self.G**2 * self.M**2 / (2 * self.c_prime**2 * r**4)
            ax -= gr_correction * x
            ay -= gr_correction * y

        # Return derivatives: [velocity in x, velocity in y, acceleration in x, acceleration in y]
        return np.array([vx, vy, ax, ay])

    def runge_kutta_4(self, t0, h, steps, include_gr=True):
        """
        Runge-Kutta 4th order method to integrate the equations of motion.

        Parameters:
        - t0 (float): Initial time
        - h (float): Time step size
        - steps (int): Number of integration steps
        - include_gr (bool): Whether to include General Relativity corrections

        Returns:
        - tuple: 
            - trajectory (np.array): Array of [x, y] positions over time
            - perihelions (list): List of perihelion distances
            - perihelion_advances (list): List of perihelion advance angles
        """
        t = t0
        state = self.initial_state
        trajectory = [state[:2]]  # Store initial position

        # Lists to track perihelions and their angles
        perihelions = []
        perihelion_angles = []
        previous_perihelion = float('inf')

        # Integrate over the specified number of steps
        for i in range(steps):
            # Compute Runge-Kutta intermediate steps (k1, k2, k3, k4)
            k1 = self.derivatives(t, state, include_gr)
            k2 = self.derivatives(t + h/2, state + h/2 * k1, include_gr)
            k3 = self.derivatives(t + h/2, state + h/2 * k2, include_gr)
            k4 = self.derivatives(t + h, state + h * k3, include_gr)

            # Update state using the weighted sum of the Runge-Kutta steps
            state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h  # Increment time by one time step

            # Store the new position in the trajectory
            trajectory.append(state[:2])

            # Calculate the distance from the central mass
            x, y = state[:2]
            r = np.sqrt(x**2 + y**2)

            # Track perihelion (minimum distance to the Sun)
            if r < previous_perihelion:
                previous_perihelion = r
            elif r > previous_perihelion:
                # Record perihelion distance and angle
                perihelions.append(previous_perihelion)
                perihelion_angle = np.arctan2(y, x)
                perihelion_angles.append(perihelion_angle)
                
                # Reset for the next perihelion search
                previous_perihelion = float('inf')

        # Calculate the advance in the perihelion angle for each orbit
        perihelion_advances = []
        for i in range(1, len(perihelion_angles)):
            delta_angle = perihelion_angles[i] - perihelion_angles[i - 1]
            
            # Adjust for angle wrap-around at 2π
            if delta_angle < 0:
                delta_angle += 2 * np.pi
            
            perihelion_advances.append(delta_angle)

        # Return trajectory, list of perihelion distances, and perihelion advances
        return np.array(trajectory), perihelions, perihelion_advances
