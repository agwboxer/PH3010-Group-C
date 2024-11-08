# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:33:17 2024

@author: H N
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants in AU units where appropriate
G = 4 * np.pi**2          # Gravitational constant in AU^3 yr^-2 M_sun^-1
M = 1                      # Mass of the Sun in Solar Masses
c = 63240                  # Speed of light in AU/year
c_prime = c / 5000         # Further reduced speed of light to amplify GR effect 

# Convert Mercury's orbital parameters to AU and years
a = 0.387                  # Semi-major axis of Mercury in AU
e = 0.2056                 # Eccentricity of Mercury's orbit
x0 = a * (1 - e)           # Start at perihelion in AU
y0 = 0
vx0 = 0
vy0 = np.sqrt(G * M * (1 + e) / (a * (1 - e)))  # Initial velocity in AU/year

# Simulation parameters
dt = 1e-5  # Time step in years
num_steps = 200000  # Increased number of steps for extended integration time

# function to calculate derivatives with GR correction
def derivatives(state, use_gr=True):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    
    # Base Newtonian terms
    accel_x = -G * M * x / r**3
    accel_y = -G * M * y / r**3
    
    # GR correction terms, only if use_gr is True
    if use_gr:
        gr_correction_x = - (3 * G * M * x) / (2 * c_prime**2 * r**4)
        gr_correction_y = - (3 * G * M * y) / (2 * c_prime**2 * r**4)
        accel_x += gr_correction_x
        accel_y += gr_correction_y
    
    return np.array([vx, vy, accel_x, accel_y])

# RK2 integration function
def rk2_step(state, dt, use_gr=True):
    k1 = derivatives(state, use_gr) * dt
    k2 = derivatives(state + 0.5 * k1, use_gr) * dt
    return state + k2

# Arrays to store the positions for plotting
x_with_gr = []
y_with_gr = []
x_without_gr = []
y_without_gr = []

# Initial state arrays [x, y, vx, vy]
state_with_gr = np.array([x0, y0, vx0, vy0])
state_without_gr = np.array([x0, y0, vx0, vy0])

# Running the simulation
for step in range(num_steps):
    # Store positions for plotting
    x_with_gr.append(state_with_gr[0])
    y_with_gr.append(state_with_gr[1])
    x_without_gr.append(state_without_gr[0])
    y_without_gr.append(state_without_gr[1])
    
    # Update the state with RK2
    state_with_gr = rk2_step(state_with_gr, dt, use_gr=True)
    state_without_gr = rk2_step(state_without_gr, dt, use_gr=False)

# Converted lists to arrays for easy plotting
x_with_gr = np.array(x_with_gr)
y_with_gr = np.array(y_with_gr)
x_without_gr = np.array(x_without_gr)
y_without_gr = np.array(y_without_gr)

# Print final Interation in both cases
print("Final state with GR correction:", state_with_gr)
print("Final state without GR correction:", state_without_gr)

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(x_with_gr, y_with_gr, label="With GR Correction", color="blue")
plt.plot(x_without_gr, y_without_gr, label="Without GR Correction", color="red", linestyle="--")
plt.xlabel("x position (AU)")
plt.ylabel("y position (AU)")
plt.title("Orbit of Mercury with and without GR Correction (RK2 Method)")
plt.legend()
plt.show()
