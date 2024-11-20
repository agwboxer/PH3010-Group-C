import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# constants and initial conditions
xi = -0.307278     # initial x position (perihelion) in AU
yi = 0.0           # initial y position in AU
vix = 0            # in AU/year
viy = 12.44        # in AU/year
M = 1.0            # mass of the Sun in solar mass units
G = 4 * np.pi**2   # gravitational constant 

# orbital parameters
T = np.sqrt(0.387**3)  # orbital period in years
T_final = 2*T        
steps = 100           
dt = T_final / steps

# acceleration function
def radius(x, y):
    return np.sqrt(x**2 + y**2)

def acceleration(x, y):
    r = radius(x, y)
    ax = -G * M * x / r**3
    ay = -G * M * y / r**3
    return np.array([ax, ay])

# define the system of ODEs 
def system_of_odes(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    dxdt = vx
    dydt = vy
    dvxdt = -G * M * x / r**3
    dvydt = -G * M * y / r**3
    return np.array([dxdt, dydt, dvxdt, dvydt])

# wrapper function to time methods
def run_with_timing(method, *args):
    start_time = time.perf_counter()
    result = method(*args)
    elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time

# Euler Method
def euler_method(xi, yi, vix, viy, dt, steps):
    xpos, ypos = [xi], [yi]
    vx, vy = vix, viy
    for _ in range(steps):
        ax, ay = acceleration(xpos[-1], ypos[-1])
        vx += ax * dt
        vy += ay * dt
        xpos.append(xpos[-1] + vx * dt)
        ypos.append(ypos[-1] + vy * dt)
    return xpos, ypos


# Leapfrog Method
def leapfrog_method(xi, yi, vix, viy, dt, steps):
    xpos, ypos = [xi], [yi]
    vx, vy = vix, viy
    for _ in range(steps):
        ax, ay = acceleration(xpos[-1], ypos[-1])
        vx_mid = vx + 0.5 * ax * dt
        vy_mid = vy + 0.5 * ay * dt
        xpos.append(xpos[-1] + vx_mid * dt)
        ypos.append(ypos[-1] + vy_mid * dt)
        ax_new, ay_new = acceleration(xpos[-1], ypos[-1])
        vx = vx_mid + 0.5 * ax_new * dt
        vy = vy_mid + 0.5 * ay_new * dt
    return xpos, ypos

# RK2 Method
def rk2_method(initial_conditions, t0, t_final, h):
    num_steps = int((t_final - t0) / h)
    x = np.zeros((num_steps + 1, 4))
    x[0] = initial_conditions
    xpos, ypos = [x[0][0]], [x[0][1]]
    for i in range(num_steps):
        k1 = h * system_of_odes(t0 + i * h, x[i])
        k2 = h * system_of_odes(t0 + (i + 0.5) * h, x[i] + 0.5 * k1)
        x[i + 1] = x[i] + k2
        xpos.append(x[i + 1][0])
        ypos.append(x[i + 1][1])
    return xpos, ypos

# RK4 Method
def rk4_method(initial_conditions, t0, t_final, h):
    steps = int((t_final - t0) / h)
    x = np.zeros((steps + 1, 4))
    x[0] = initial_conditions
    xpos, ypos = [x[0][0]], [x[0][1]]
    for i in range(steps):
        k1 = h * system_of_odes(t0 + i * h, x[i])
        k2 = h * system_of_odes(t0 + (i + 0.5) * h, x[i] + 0.5 * k1)
        k3 = h * system_of_odes(t0 + (i + 0.5) * h, x[i] + 0.5 * k2)
        k4 = h * system_of_odes(t0 + (i + 1) * h, x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        xpos.append(x[i + 1][0])
        ypos.append(x[i + 1][1])
    return xpos, ypos

# solve the system using solve_ivp using RK23
solution, rk23_time = run_with_timing(solve_ivp, system_of_odes, [0, T_final], [xi, yi, vix, viy], 
                                      'RK23', np.linspace(0, T_final, steps))
x_true, y_true = solution.y[0], solution.y[1]

# execute the other methods
(x_euler, y_euler), euler_time = run_with_timing(euler_method, xi, yi, vix, viy, dt, steps)
(x_leapfrog, y_leapfrog), leapfrog_time = run_with_timing(leapfrog_method, xi, yi, vix, viy, dt, steps)
(x_rk2, y_rk2), rk2_time = run_with_timing(rk2_method, [xi, yi, vix, viy], 0, T_final, dt)
(x_rk4, y_rk4), rk4_time = run_with_timing(rk4_method, [xi, yi, vix, viy], 0, T_final, dt)

# print timings for all methods
print(f"RK23 time: {rk23_time:.6f} seconds")
print(f"Euler time: {euler_time:.6f} seconds")
print(f"Leapfrog time: {leapfrog_time:.6f} seconds")
print(f"RK2 time: {rk2_time:.6f} seconds")
print(f"RK4 time: {rk4_time:.6f} seconds")

# plots for the orbits for comparison
plt.figure(figsize=(14, 10))
plt.plot(0, 0, 'yo', label="Sun", markersize=10)
plt.plot(x_true, y_true, 'k', label='RK23', linewidth=2)
plt.plot(x_euler, y_euler, 'r--', label='Euler Method')
plt.plot(x_leapfrog, y_leapfrog, 'g--', label='Leapfrog Method')
plt.plot(x_rk2, y_rk2, 'b--', label='RK2 Method')
plt.plot(x_rk4, y_rk4, 'm--', label='RK4 Method')
plt.xlabel('x position (AU)')
plt.ylabel('y position (AU)')
plt.legend()
plt.title('Orbit of Mercury around the Sun - Comparison of Numerical Methods with RK23')
plt.grid()
plt.show()
