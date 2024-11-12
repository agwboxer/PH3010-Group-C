"""
Created on Fri Nov  1 14:43:24 2024

@author: oliver
"""

import numpy as np
import matplotlib.pyplot as plt

#initial conditions, all initial values taken at the perihelion in standard astronomical units
Ecc = 0.2056       #Eccentricity of orbit (mercury)
a = 0.387           # Semi major distance
b = a*np.sqrt(1-Ecc**2)     #Semi minor
Tsq = a**3          # Orbit Period squared
T = np.sqrt(Tsq)    # Orbit period

peri_d = a-Ecc*a
aphe_d = a+Ecc*a


viy_calc = np.sqrt(((2*4*np.pi**2)*(1/peri_d - 1/aphe_d))/(1-(peri_d/aphe_d)**2))   #Initial velocity from derivation from energy and angular momentum


xi = -(a-Ecc*a)     # inital x position at periastron, taken as the semi major of mercurys orbit minus this value times the orbital eccentricity
yi = 0.0            # Initial y value
xs = 0.0            # x position of sun
ys = 0.0            # y position of sun
vix = 0             # Initial x velocity of mercury 
viy = viy_calc          # Initial y velocity of mercury 

Ms = 1.0            # Mass of the sun in solar mass units 
G = 4*np.pi**2      # Gravitational constant G 

steps = 100     # Number of steps plotted over, an increase in the number of steps makes the orbit more and more correct, and the orbits begins to overlap with itself
dt = T/steps        # Length of a step defined by the period and how many steps there are 
t = np.arange(0.0, T, dt)   #array of each step up to the period

def radius(x, y):
    """
    Parameters
    ----------
    x : x position of mercury
    y : y position of mercury
    Returns
    -------
    current radius of the star
    """
    return np.sqrt(x**2 + y**2)

# Define the system of ODEs
def system_of_odes(state, t):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2) 
    dxdt = vx  
    dydt = vy  
    dvxdt = -G * Ms * x / r**3  
    dvydt = -G * Ms * y / r**3  
    return np.array([dxdt, dydt, dvxdt, dvydt])

def Euler(initial_conditions, t0, t_final, dt):
    num_steps = int((t_final - t0) / dt)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))  
    x[0] = initial_conditions
    
    #Euler Loop
    for i in range(num_steps):
        k1 = dt * system_of_odes(x[i], t[i])  
        x[i + 1] = x[i] + k1
    
    return t, x

def RK2(initial_conditions, t0, t_final, h):
    # Number of steps
    num_steps = int((t_final - t0) / h)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))  
    x[0] = initial_conditions 
    
    # RK2 method loop
    for i in range(num_steps):
        k1 = h * system_of_odes(x[i], t[i])  
        k2 = h * system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * h) 
        x[i + 1] = x[i] + k2  
    
    return t, x

def RK4(initial_conditions, t0, t_final, dt):
    # Number of steps
    num_steps = int((t_final - t0) / dt)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))  
    x[0] = initial_conditions 
    
    # RK4 method loop
    for i in range(num_steps):
        k1 = dt * system_of_odes(x[i], t[i])  
        k2 = dt * system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
        k3 = dt * system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
        k4 = dt * system_of_odes(x[i] + k3, t[i] + 0.5 * dt)
        x[i + 1] = x[i] + k1/6 + k2/3 + k3/3 + k4/6 
    
    return t, x

initial_conditions = [xi , yi, vix, viy]  

# Solve the system using Euler, RK2 and RK4
steps_data = np.linspace(10, 500, 50)

store_E_dist = []
store_RK2_dist = []
store_RK4_dist = []

store_t_data = []

for i, val in enumerate(steps_data):
    dt = T/steps_data[i]

    Eulert, EulerSoln = Euler(initial_conditions, 0, T, dt)    
    final_EData = EulerSoln[-1]
    distance_E = np.sqrt((-a-final_EData[0])**2 + final_EData[1]**2)    
    store_E_dist = np.append(store_E_dist, distance_E)
    
    RK2t, RK2Soln = RK2(initial_conditions, 0, T, dt)
    final_RK2Data = RK2Soln[-1]
    distance_RK2 = np.sqrt((-a-final_RK2Data[0])**2 + final_RK2Data[1]**2)
    store_RK2_dist = np.append(store_RK2_dist, distance_RK2)
    
    RK4t, RK4Soln = RK4(initial_conditions, 0, T, dt)
    final_RK4Data = RK4Soln[-1]
    distance_RK4 = np.sqrt((-a-final_RK4Data[0])**2 + final_RK4Data[1]**2)
    store_RK4_dist = np.append(store_RK4_dist, distance_RK4)
    
    store_t_data = np.append(store_t_data, steps_data[i])
    
plt.figure()
plt.scatter(store_t_data, store_E_dist, marker='.')
plt.scatter(store_t_data, store_RK2_dist, marker='.')
plt.scatter(store_t_data, store_RK4_dist, marker='.')
plt.xlabel('No. of timesteps')
plt.ylabel('Distance from start (AU)')
plt.title('How time steps affect where the orbit ends relative to a fixed starting position')
