import matplotlib.pyplot as plt
from Newtonian_Simulation import Newtonian_Orbit
import numpy as np



def calculate_energies(sol):
    """
    Calculate kinetic and potential energy from the simulation results

    Parameters:
    - sol (array): Array containing [x, y, vx, vy] from the simulation

    Returns:
    - K (array): Kinetic energy as a function of time
    - U (array): Potential energy as a function of time
    - E (array): Total energy as a function of time
    """
    M_m = 1.6595 * 10 ** -7 # Mass of Mercury in solar mass units
    x, y, vx, vy = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    r = np.sqrt(x**2 + y**2)  # Distance from the Sun
    K = 0.5 * M_m * (vx**2 + vy**2)  # Kinetic energy
    U = -Newtonian_Orbit.G * Newtonian_Orbit.Ms * M_m / r  # Potential energy
    E = K + U  # Total energy
    return K, U, E



