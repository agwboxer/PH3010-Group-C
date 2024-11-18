# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:19:54 2024

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt

G = 39.478  
M = 1
c_prime = 63.239  # Modified speed of light (c/100)

x0 = 0.307  
y0 = 0      
vx0 = 0     
vy0 = 12.4375  

def derivatives(t, state, include_gr=True):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)  
    
    # Newtonian accelerations
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3
    
    # GR correction term
    if include_gr:
        gr_correction = -3 * G**2 * M**2 / (2 * c_prime**2 * r**4)
        ax -= gr_correction * x
        ay -= gr_correction * y
    
    return np.array([vx, vy, ax, ay])

def runge_kutta_4(derivatives, t0, initial_state, h, steps, include_gr=True):
    t = t0
    state = initial_state
    trajectory = [state[:2]]  
    
    perihelions = []  
    perihelion_angles = []  
    previous_perihelion = float('inf')  
    
    for i in range(steps):
        k1 = derivatives(t, state, include_gr)
        k2 = derivatives(t + h/2, state + h/2 * k1, include_gr)
        k3 = derivatives(t + h/2, state + h/2 * k2, include_gr)
        k4 = derivatives(t + h, state + h * k3, include_gr)
        
        # Update the state using the Runge-Kutta method
        state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        trajectory.append(state[:2])
    
        x, y = state[:2]
        r = np.sqrt(x**2 + y**2)
        
        # Track perihelion (minimum distance to the Sun)
        if r < previous_perihelion:
            previous_perihelion = r
        elif r > previous_perihelion:
            perihelions.append(previous_perihelion)
            
            perihelion_angle = np.arctan2(y, x)  
            perihelion_angles.append(perihelion_angle)
            
            previous_perihelion = float('inf')  
    
    # Calculate perihelion advance
    perihelion_advances = []
    for i in range(1, len(perihelion_angles)):
        delta_angle = perihelion_angles[i] - perihelion_angles[i - 1]
        
        if delta_angle < 0:
            delta_angle += 2 * np.pi
        
        perihelion_advances.append(delta_angle)  
    
    return np.array(trajectory), perihelions, perihelion_advances

initial_state = np.array([x0, y0, vx0, vy0])

t0 = 0
t_end = 10
h = 1/365  
steps = int((t_end - t0) / h)

trajectory, perihelions, perihelion_advances = runge_kutta_4(derivatives, t0, initial_state, h, steps, include_gr=True)

plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Mercury orbit (with GR correction)')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Orbit of Mercury around the Sun')
plt.grid()
plt.legend()
plt.axis('equal')
plt.show()

if len(perihelion_advances) > 0:
    average_advance = np.mean(perihelion_advances)  
    print(f"Average perihelion advance per revolution: {average_advance:.6f} radians")
else:
    print("Not enough perihelion data to calculate advance.")












        

        
        
        
    
    


    
