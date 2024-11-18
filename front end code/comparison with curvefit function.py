author @carolena
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# constants and initial conditions for orbit
a = 1.0  # Semi-major axis (AU)
Ms = 1.0  # Mass of the Sun in solar mass units
G = 4 * np.pi**2  # Gravitational constant in AU^3 / (days^2 * Ms)
T = np.sqrt(a**3 / (G * Ms))  # Orbital period in days
xi, yi = a, 0  # Initial position of Mercury at (1, 0) in AU
vix, viy = 0, np.sqrt(G * Ms / a)  # Initial velocity for circular orbit

# time parameters
steps_data = np.linspace(10, 500, 50)  # Different timesteps for testing
store_error = []  # To store the global errors

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
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))
    x[0] = initial_conditions
    
    for i in range(num_steps):
        k1 = dt * system_of_odes(x[i], t[i])
        x[i + 1] = x[i] + k1
    
    return t, x

# Define the RK4 method for solving ODEs
def RK4(initial_conditions, t0, t_final, dt):
    num_steps = int((t_final - t0) / dt)
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))
    x[0] = initial_conditions
    
    for i in range(num_steps):
        k1 = dt * system_of_odes(x[i], t[i])
        k2 = dt * system_of_odes(x[i] + 0.5 * k1, t[i] + 0.5 * dt)
        k3 = dt * system_of_odes(x[i] + 0.5 * k2, t[i] + 0.5 * dt)
        k4 = dt * system_of_odes(x[i] + k3, t[i] + dt)
        x[i + 1] = x[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, x

# Initial conditions for the orbital motion
initial_conditions = [xi, yi, vix, viy]


# Loop over different step sizes to compute the error
for i, val in enumerate(steps_data):
    dt = T / steps_data[i]
    
    # Solve using Euler method
    Eulert, EulerSoln = Euler(initial_conditions, 0, T, dt)
    final_Euler = EulerSoln[-1]
    error_Euler = np.abs(np.sqrt(final_Euler[0]**2 + final_Euler[1]**2) - a) #take abs to avoid negative plot
    
    # Solve using RK4 method
    RK4t, RK4Soln = RK4(initial_conditions, 0, T, dt)
    final_RK4 = RK4Soln[-1]
    error_RK4 = np.abs(np.sqrt(final_RK4[0]**2 + final_RK4[1]**2) - a)
    
    # Store the error for both methods
    store_error.append((steps_data[i], error_Euler, error_RK4))

# Convert store_error to an array 
store_error = np.array(store_error)

# Plot the errors for different time steps
plt.figure()
plt.loglog(store_error[:, 0], store_error[:, 1], label='Euler Error', marker='.')
plt.loglog(store_error[:, 0], store_error[:, 2], label='RK4 Error', marker='.')
plt.xlabel('Step size ')
plt.ylabel('Global Error (AU)')
plt.title('Convergence plot')
plt.show()

# map the error data to a power law for Euler and RK4
def power_law(x, k, n):
    return k * x**n

# Fit the error data to a power law for both 
params_Euler, _ = curve_fit(power_law, store_error[:, 0], store_error[:, 1])
params_RK4, _ = curve_fit(power_law, store_error[:, 0], store_error[:, 2])

# printed results
print("Euler Method: k =", params_Euler[0], ", n =", params_Euler[1])
print("RK4 Method: k =", params_RK4[0], ", n =", params_RK4[1])

#not sure why final parameters are negative must match decently to predicted numerical integration techniques 
