import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
viy = 12.0          # Initial y velocity of mercury 

Ms = 1.0            # Mass of the sun in solar mass units 
G = 4*np.pi**2      # Gravitational constant G 


steps = 5000      # Number of steps plotted over, an increase in the number of steps makes the orbit more and more correct, and the orbits begins to overlap with itself
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

def Euler(initial_conditions, t0, t_final, dt):
    num_steps = int((t_final - t0) / dt)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))  
    x[0] = initial_conditions
    
    for i in range(num_steps):
        k1 = dt * system_of_odes(x[i], t[i])  
        x[i + 1] = x[i] + k1
    
    return t, x

def RK4(initial_conditions, t0, t_final, dt):
    # Number of steps
    num_steps = int((t_final - t0) / dt)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 4))  
    x[0] = initial_conditions 

    # RK2 method loop
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
plt.plot(store_t_data, store_E_dist, marker='.')
plt.plot(store_t_data, store_RK2_dist, marker='.')
plt.plot(store_t_data, store_RK4_dist, marker='.')
plt.xlabel('No. of timesteps')
plt.ylabel('Distance from start (AU)')
plt.title('How time steps affect where the orbit ends relative to a fixed starting position')
    
# reference solution to find convergence of Euler’s method using RK4 with very small step size
ref_steps = 100000
ref_dt = T / ref_steps
_, ref_solution = RK4(initial_conditions, 0, T, ref_dt)
ref_x, ref_y = ref_solution[-1, 0], ref_solution[-1, 1]
ref_pos = np.sqrt(ref_x**2 + ref_y**2)

# global errors for each method
global_error_euler = []
global_error_rk2 = []
global_error_rk4 = []
step_sizes = []

for val in steps_data:
    dt = T / val
    step_sizes.append(dt)

   
    _, euler_sol = Euler(initial_conditions, 0, T, dt)
    euler_x, euler_y = euler_sol[-1, 0], euler_sol[-1, 1]
    euler_pos = np.sqrt(euler_x**2 + euler_y**2)
    global_error_euler.append(abs(euler_pos - ref_pos))

   
    _, rk2_sol = RK2(initial_conditions, 0, T, dt)
    rk2_x, rk2_y = rk2_sol[-1, 0], rk2_sol[-1, 1]
    rk2_pos = np.sqrt(rk2_x**2 + rk2_y**2)
    global_error_rk2.append(abs(rk2_pos - ref_pos))

    
    _, rk4_sol = RK4(initial_conditions, 0, T, dt)
    rk4_x, rk4_y = rk4_sol[-1, 0], rk4_sol[-1, 1]
    rk4_pos = np.sqrt(rk4_x**2 + rk4_y**2)
    global_error_rk4.append(abs(rk4_pos - ref_pos))

# Plot global errors on a log-log plot
plt.figure()
plt.loglog(step_sizes, global_error_euler, marker=".")
plt.loglog(step_sizes, global_error_rk2, marker=".")
plt.loglog(step_sizes, global_error_rk4, marker=".")
plt.xlabel("step size ")
plt.ylabel("global error")
plt.title("convergence plot")
plt.show()



