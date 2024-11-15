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
xi_v, yi_v = -(aV-EccV*aV) , 0.0

# --------------------  MASSES  -------------------- #

Ms_kg = 1.988416*10**30
Ms = 1.0            # Mass of the sun in solar mass units 
Mm = 0.33010*10**24 / Ms_kg
Mv = 4.8673*10**24 / Ms_kg  #954.79194*10**(-6)
G = 4*np.pi**2      # Gravitational constant G 


# --------------------  TIMINGS  -------------------- #

Tsq = a**3          # Orbit Period squared
T = np.sqrt(Tsq)    # Orbit period
Factor = 30         # Factor used to scale the total time for the whole code
steps = 750       # Number of steps plotted over, an increase in the number of steps makes the orbit more and more correct, and the orbits begins to overlap with itself
dt = T/steps        # Length of a step defined by the period and how many steps there are 
t = np.arange(0.0, Factor*T, dt)   #array of each step up to the period


# --------------------  ARRAYS  -------------------- #

#here the previously defined data is collected into arrays for use in future code and functions
initial_position_merc = [xi, yi]
inital_position_ven =  [xi_v, yi_v]

initial_velocity_merc = [vix, viy]
initial_velocity_ven = [0, viy_calcV]

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

    df1_dt = ((G*ms*(ps - p1))/(np.linalg.norm(ps - p1)**3) + (G*m2*(p2 - p1))/(np.linalg.norm(p2 - p1)**3))
    df2_dt = ((G*ms*(ps - p2))/(np.linalg.norm(ps - p2)**3) + (G*m1*(p1 - p2))/(np.linalg.norm(p1 - p2)**3))

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()


c_prime = 63.239
def system_of_odes_GR(t, S, m1, m2):
    #Sun position and mass
    ms = 1.0
    ps = [0, 0]
    # p1, p2, p3, dp1_dt, dp2_dt, dp3_dt = S
    p1, p2 = S[0:2], S[2:4]
    dp1_dt, dp2_dt = S[4:6], S[6:8]

    f1, f2 = dp1_dt, dp2_dt

    df1_dt = (((G*ms*(ps - p1))/(np.linalg.norm(ps - p1)**3) + (G*m2*(p2 - p1))/(np.linalg.norm(p2 - p1)**3)) + ((3*(G**2)*(ms**2)*(ps - p1))/(2*(c_prime**2)*np.linalg.norm(ps - p1)**4) + (3*(G**2)*(m2**2)*(p2 - p1))/(2*(c_prime**2)*np.linalg.norm(p2 - p1)**4)))
    df2_dt = (((G*ms*(ps - p2))/(np.linalg.norm(ps - p2)**3) + (G*m1*(p1 - p2))/(np.linalg.norm(p1 - p2)**3)) + ((3*(G**2)*(ms**2)*(ps - p2))/(2*(c_prime**2)*np.linalg.norm(ps - p2)**4) + (3*(G**2)*(m1**2)*(p1 - p2))/(2*(c_prime**2)*np.linalg.norm(p1 - p2)**4)))

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()



# # splve the system of ODEs with the scipy function solve.
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

# --------------------  RK4  -------------------- #

#Compiutes the orbit with GR correction using RK4
def RK4(initial_conditions, t0, t_final, dt, Mv, Mm):
    # Number of steps
    num_steps = int((t_final - t0) / dt)
    # Time and solution arrays
    t = np.linspace(t0, t_final, num_steps + 1)
    x = np.zeros((num_steps + 1, 8))  
    x[0] = initial_conditions 
    
    # RK4 method loop
    for i in range(num_steps):
        k1 = dt * system_of_odes_GR(t[i], x[i], Mm, Mv)  
        k2 = dt * system_of_odes_GR(t[i] + 0.5 * dt, x[i] + 0.5 * k1, Mm, Mv)
        k3 = dt * system_of_odes_GR(t[i] + 0.5 * dt, x[i] + 0.5 * k2, Mm, Mv)
        k4 = dt * system_of_odes_GR(t[i] + 0.5 * dt, x[i] + k3, Mm, Mv)
        x[i + 1] = x[i] + k1/6 + k2/3 + k3/3 + k4/6 
    
    return t, x

# Solve the system using Euler, RK2 and RK4
RK4t, RK4Soln = RK4(initial_conditions_new, 0, Factor*T, dt, Mv, Mm)
RK4tB, RK4SolnB = RK4(initial_conditions_new, 0, Factor*T, dt, 0, 0)



# -------------------- GR -------------------- #

# #Computes the orbit with GR correction using scipy
# #To use this, unhash this section and hash the previous soloution section
# solutionGR = solve_ivp(
#     fun = system_of_odes_GR,
#     t_span = (0, Factor*T),
#     y0 = initial_conditions_new,
#     t_eval = t,
#     args = (Mm, Mv),
#     method='RK23'     #RK23 replaces the standard RK45 when method is unspecified
#     )

# t_sol = solutionGR.t
# mercx_sol = solutionGR.y[0]
# mercy_sol = solutionGR.y[1]

# venx_sol = solutionGR.y[2]
# veny_sol = solutionGR.y[3]

# sunx_sol = solutionGR.y[4]
# suny_sol = solutionGR.y[5]


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

# ------------  Solid Line Plots  ------------ #

# Solid line plots can be unhashed for a more simple or quick view

# # Plotted via scipy solver, using method specified previously
# ax.plot(mercx_sol, mercy_sol, [0] * len(mercx_sol), 'green', label='Mercury', linewidth=1)
# ax.plot(venx_sol, veny_sol, [0] * len(venx_sol), 'red', label='Venus', linewidth=1)

# #Plots using RK4 with GR correction
plt.plot(RK4Soln[:, 0], RK4Soln[:, 1], label="RK4 Orbit Mercury", color='Orange') 
plt.plot(RK4Soln[:, 2], RK4Soln[:, 3], label="RK4 Orbit Venus", color='Purple')

# #Plots using RK4 the orbit of mercury and venus supposing there is no force applied between each other
# plt.plot(RK4SolnB[:, 0], RK4SolnB[:, 1], label="RK4 Orbit Mercury No Venus", color='Blue') 
# plt.plot(RK4SolnB[:, 2], RK4SolnB[:, 3], label="RK4 Orbit Venus No Mercury", color='Green')

# Plot the sun
sun_dot, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=6, label='Sun')
plt.legend()

# ------------  Animation  ------------ #

# #Animation code below can be hashed or unhashed as needed
# Create animated lines and dots (these will be updated in the animation)

# #Animation using the solver function
# merc_plt, = ax.plot([], [], [], 'green', linewidth=1, label='Mercury')
# ven_plt, = ax.plot([], [], [], 'red', linewidth=1, label='Venus')
# merc_dot, = ax.plot([mercx_sol[-1]], [mercy_sol[-1]], [0], 'o', color='green', markersize=6)
# ven_dot, = ax.plot([venx_sol[-1]], [veny_sol[-1]], [0], 'o', color='red', markersize=6)
# sun_dot, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=6, label='Sun')


# #Animation using the RK4
# merc_plt, = ax.plot([], [], [], 'green', linewidth=1, label='Mercury')
# ven_plt, = ax.plot([], [], [], 'red', linewidth=1, label='Venus')
# merc_dot, = ax.plot([RK4Soln[:, 0][-1]], [RK4Soln[:, 1][-1]], [0], 'o', markersize=6, color='green') 
# ven_dot, = ax.plot([RK4Soln[:, 2][-1]], [RK4Soln[:, 3][-1]], [0], 'o', markersize=6, color='red')
# sun_dot, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=6, label='Sun')

# #Set plot details
# ax.set_title("2 Planets Orbit Around the Sun")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.zlim = (-0.5, 0.5)
# plt.legend()

#Animation function
# def update(frame):
#     # lower_lim = max(0, frame - 300)  # hash in if dont want full lines trailing behind the animation,then add lower_lim before the colons in the next code lines
#     print(f"Progress: {(frame+1)/len(t):.1%}", end='\r')

#     # Current position slices for animation
#     x_current_1 = mercx_sol[:frame+1]
#     y_current_1 = mercy_sol[:frame+1]
#     z_current_1 = [0] * len(x_current_1)  # assuming z=0

#     x_current_2 = venx_sol[:frame+1]
#     y_current_2 = veny_sol[:frame+1]
#     z_current_2 = [0] * len(x_current_2)  # assuming z=0

#     # Update animated lines and dots
#     merc_plt.set_data_3d(x_current_1, y_current_1, z_current_1)
#     merc_dot.set_data_3d([x_current_1[-1]], [y_current_1[-1]], [z_current_1[-1]])

#     ven_plt.set_data_3d(x_current_2, y_current_2, z_current_2)
#     ven_dot.set_data_3d([x_current_2[-1]], [y_current_2[-1]], [z_current_2[-1]])

#     return merc_plt, merc_dot, ven_plt, ven_dot

# # Create animation
# animation = FuncAnimation(fig, update, frames=range(0, len(t), 5), interval=5, blit=True)
plt.show()



#inspired by and based on https://www.youtube.com/watch?v=FXkH9-4EN_8

