# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:59:10 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 39.478 
M = 1
x0 = 0.307  
y0 = 0.0  
vx0 = 0.0  
vy0 = 12.4375

# Function to compute derivatives
def derivatives(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)  # Distance from the Sun
    
    # Compute acceleration
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3
    
    return np.array([vx, vy, ax, ay])

# Runge-Kutta 4th Order Method
def runge_kutta_4(func, t0, y0, h, steps):
    t = t0
    y = y0
    trajectory = [y0[:2]]  # Store positions (x, y)

    for _ in range(steps):
        k1 = func(t, y)
        k2 = func(t + h / 2, y + h / 2 * k1)
        k3 = func(t + h / 2, y + h / 2 * k2)
        k4 = func(t + h, y + h * k3)

        y = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        trajectory.append(y[:2])  # Store positions (x, y)
    
    return np.array(trajectory)

# Initial conditions
initial_state = np.array([x0, y0, vx0, vy0])

t0 = 0.0          
t_end = 0.5
h = 1/365
steps = int((t_end - t0) / h)


current_trajectory = runge_kutta_4(derivatives, t0, initial_state, h, steps)
    
# Plotting the trajectory
plt.figure()
plt.plot(current_trajectory[:, 0], current_trajectory[:, 1], label=f'Time step h = {h*365} days')

plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Orbit of Mercury Around the Sun')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend(fontsize='small', loc='upper right')
plt.axis('equal')
plt.show()

#calculations
x_positions = current_trajectory[:, 0]  # Extract x positions
y_positions = current_trajectory[:, 1]  # Extract y positions
    
distances = np.sqrt(x_positions**2 + y_positions**2)
    
perihelion = np.min(distances)
aphelion = np.max(distances)

semi_major_axis = (perihelion + aphelion) / 2

print(f"Time step h = {h*365:.0f} days: Aphelion = {aphelion:.3f} AU, Perihelion = {perihelion:.3f} AU, Semi-major axis = {semi_major_axis:.3f} AU")


    
    
    