"""
Created on Fri Nov  1 14:43:24 2024

@author: oliver
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#initial conditions, all initial values taken at the perihelion in standard astronomical units


# --------------------  MERCURY  -------------------- #
Ecc, a = 0.2056, 0.387               # Eccentricity of orbit (mercury), Semi major distance
b = a*np.sqrt(1-Ecc**2)              # Semi minor
peri_d, aphe_d = a-Ecc*a, a+Ecc*a
viy_calc = np.sqrt(((2*4*np.pi**2)*(1/peri_d - 1/aphe_d))/(1-(peri_d/aphe_d)**2))   # Initial velocity from derivation from energy and angular momentum

xi, yi = -(a-Ecc*a) , 0.0            # inital x / y position at periastron, taken as the semi major of mercurys orbit minus this value times the orbital eccentricity
vix, viy = 0.0, viy_calc             # Initial x / y velocity of mercury 


# --------------------  VENUS  -------------------- #

EccV, aV = 0.0068, 0.723             # The eccentricity and semi major distance of venus
peri_dV, aphe_dV  = aV-EccV*aV, aV+EccV*aV
viy_calcV = np.sqrt(((2*4*np.pi**2)*(1/peri_dV - 1/aphe_dV))/(1-(peri_dV/aphe_dV)**2))


# --------------------  MASSES  -------------------- #

Ms_kg = 1.989*10**30
Ms = 1.0            # Mass of the sun in solar mass units 
Mm = 0.33*10**24 / Ms_kg
Mv = 4.87*10**24 / Ms_kg  #954.79194*10**(-6)
G = 4*np.pi**2      # Gravitational constant G 


# --------------------  TIMINGS  -------------------- #

Tsq = a**3          # Orbit Period squared
T = np.sqrt(Tsq)    # Orbit period
Factor = 15
steps = 1000        # Number of steps plotted over, an increase in the number of steps makes the orbit more and more correct, and the orbits begins to overlap with itself
dt = T/steps        # Length of a step defined by the period and how many steps there are 
t = np.arange(0.0, Factor*T, dt)   #array of each step up to the period


# --------------------  ARRAYS  -------------------- #

#here the previously defined data is collected into arrays for use in future code and functions
initial_position_merc = [xi, yi]
inital_position_ven =  [-aV, 0.0]
initial_position_sun = [0, 0]

initial_velocity_merc = [vix, viy]
initial_velocity_ven = [0, viy_calcV]
initial_velocity_sun = [0, 0]

initial_conditions_new = np.array([
    initial_position_merc, inital_position_ven,
    initial_velocity_merc, initial_velocity_ven
    ]).ravel()


# Define the system of ODEs
def system_of_odes(t, S, m1, m2):
    ms = 1.0
    ps = [0, 0]
    # p1, p2, p3, dp1_dt, dp2_dt, dp3_dt = S
    p1, p2 = S[0:2], S[2:4]
    dp1_dt, dp2_dt = S[4:6], S[6:8]

    f1, f2 = dp1_dt, dp2_dt

    df1_dt = (G*ms*(ps - p1))/(np.linalg.norm(ps - p1)**3) + (G*m2*(p2 - p1))/(np.linalg.norm(p2 - p1)**3)
    df2_dt = (G*ms*(ps - p2))/(np.linalg.norm(ps - p2)**3) + (G*m1*(p1 - p2))/(np.linalg.norm(p1 - p2)**3)

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()


# splve the system of ODEs with the scipy function solve.
solution = solve_ivp(
    fun = system_of_odes,
    t_span = (0, Factor*T),
    y0 = initial_conditions_new,
    t_eval = t,
    args = (Mm, Mv),
    method='RK23'     #RK23 replaces the standard RK45 when method is unspecified
    )

# set the soloution to variables
t_sol = solution.t
mercx_sol = solution.y[0]
mercy_sol = solution.y[1]

venx_sol = solution.y[2]
veny_sol = solution.y[3]

sunx_sol = solution.y[4]
suny_sol = solution.y[5]

# --------------------  2D Plot  -------------------- #

# fig, ax2 = plt.subplots()
# ax2.plot(mercx_sol, mercy_sol, 'green', label='Mercury', linewidth=1)
# ax2.plot(venx_sol, veny_sol, 'red', label='Venus', linewidth=1)
# sun_dot, = ax2.plot([0], [0], 'o', color='blue', markersize=6, label='Sun')
# ax2.set_title("2 Planets Orbit Around the Sun")
# ax2.set_xlabel("x")
# ax2.set_ylabel("y")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.legend()


# --------------------  3D Plots  -------------------- #

#Start a 3D plot as it can show animation very nicely, even though this problem is confined in the 2D planetary plane
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Solid line plots can be unhashed for a more simple or quick view
# ax.plot(mercx_sol, mercy_sol, [0] * len(mercx_sol), 'green', label='Mercury', linewidth=1)
# ax.plot(venx_sol, veny_sol, [0] * len(venx_sol), 'red', label='Venus', linewidth=1)

# Create animated lines and dots (these will be updated in the animation)
merc_plt, = ax.plot([], [], [], 'green', linewidth=1, label='Mercury')
ven_plt, = ax.plot([], [], [], 'red', linewidth=1, label='Venus')
merc_dot, = ax.plot([mercx_sol[-1]], [mercy_sol[-1]], [0], 'o', color='green', markersize=6)
ven_dot, = ax.plot([venx_sol[-1]], [veny_sol[-1]], [0], 'o', color='red', markersize=6)
sun_dot, = ax.plot([0], [0], [0], 'o', color='blue', markersize=6, label='Sun')

ax.set_title("2 Planets Orbit Around the Sun")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.zlim = (-0.5, 0.5)
plt.legend()

# Can hash out everything below here if you dont want the animation and just want the plot
def update(frame):
    # lower_lim = max(0, frame - 300)  # hash in if dont want full lines trailing behind the animation,then add lower_lim before the colons in the next code lines
    print(f"Progress: {(frame+1)/len(t):.1%}", end='\r')

    # Current position slices for animation
    x_current_1 = mercx_sol[:frame+1]
    y_current_1 = mercy_sol[:frame+1]
    z_current_1 = [0] * len(x_current_1)  # assuming z=0

    x_current_2 = venx_sol[:frame+1]
    y_current_2 = veny_sol[:frame+1]
    z_current_2 = [0] * len(x_current_2)  # assuming z=0

    # Update animated lines and dots
    merc_plt.set_data_3d(x_current_1, y_current_1, z_current_1)
    merc_dot.set_data_3d([x_current_1[-1]], [y_current_1[-1]], [z_current_1[-1]])

    ven_plt.set_data_3d(x_current_2, y_current_2, z_current_2)
    ven_dot.set_data_3d([x_current_2[-1]], [y_current_2[-1]], [z_current_2[-1]])

    return merc_plt, merc_dot, ven_plt, ven_dot

# Create animation
animation = FuncAnimation(fig, update, frames=range(0, len(t), 2), interval=5, blit=True)
plt.show()



#inspired by and based on https://www.youtube.com/watch?v=FXkH9-4EN_8
