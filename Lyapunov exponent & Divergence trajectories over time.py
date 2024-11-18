# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:03:26 2024

@author: Harry 

Added changes: Corrected the lyapunov exponent calculation, 
made it so that the curve fits for 1 mercury year instead of `1 earth year. 
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define Mercury's orbit equations in terms of position and velocity
def mercury_orbit(t, state):
    x, vx, y, vy = state
    r = np.sqrt(x**2 + y**2)
    # Gravitational constant times mass of the Sun (in AU^3/yr^2)
    GM = 4 * np.pi**2
    # Equations of motion
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    return [vx, ax, vy, ay]

# Initial conditions for Mercury's orbit (position in AU, velocity in AU/year)
initial_state = [0.39, 0, 0, 9.96]  # Mercury's average orbital radius and velocity

# Time span for one full orbit (88 days = 0.2407 years)
time_span = (0, 0.2407)  # 88 days in years
time_eval = np.linspace(0, 0.2407, 1000)  # Time evaluation points

# Solving the orbit for the original trajectory
solution_original = solve_ivp(mercury_orbit, time_span, initial_state, t_eval=time_eval)

# I defined a small perturbation to initial conditions
perturbation = 1e-5  # Small change in initial velocity

# Initial conditions for the perturbed orbit
initial_state_perturbed = [0.39, 0, 0, 9.96 + perturbation]

# Solving for the perturbed orbit
solution_perturbed = solve_ivp(mercury_orbit, time_span, initial_state_perturbed, t_eval=time_eval)

# Calculating the separation (distance) between the two trajectories over time
x_diff = solution_perturbed.y[0] - solution_original.y[0]
y_diff = solution_perturbed.y[2] - solution_original.y[2]
separation = np.sqrt(x_diff**2 + y_diff**2)

# Adding a small threshold to avoid log(0) or very small separations causing issues
threshold = 1e-10  # Small value to avoid division by zero
separation = np.maximum(separation, threshold)  # Ensuring separation never goes below threshold

# Calculating the Lyapunov exponent using logarithmic growth of separation
log_separation = np.log(separation)

# Fit a straight line to the log of the separation over time to estimate the Lyapunov exponent
# The slope of this line is the Lyapunov exponent
slope, intercept = np.polyfit(time_eval, log_separation, 1)
lyapunov_exponent = slope  # The slope is the Lyapunov exponent

# Plotting the separation over time (Divergence)
plt.figure(figsize=(8, 6))
plt.plot(time_eval, separation, label="Separation (Perturbation)")
plt.yscale("log")
plt.title("Divergence of Trajectories Over 1 year")
plt.xlabel("Time (years)")
plt.ylabel("Separation (AU)")
plt.legend()
plt.grid()
plt.show()

# Plotting the 2D phase space: x position vs. x velocity for both trajectories
plt.figure(figsize=(8, 6))
plt.plot(solution_perturbed.y[0], solution_perturbed.y[1], color="red", label="Perturbed Trajectory")
plt.title("2D Phase Space: x Position vs. x Velocity Over 1 year")
plt.xlabel("x Position (AU)")
plt.ylabel("x Velocity (AU/year)")
plt.legend()
plt.grid()
plt.show()

# Output the estimated Lyapunov exponent (the slope of the log(separation) vs. time plot)
print(f"Estimated Lyapunov Exponent: {lyapunov_exponent:.6f} (1/year)")
