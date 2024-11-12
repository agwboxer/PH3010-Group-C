import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
vix = 0.0             # Initial x velocity of mercury 
viy = viy_calc          # Initial y velocity of mercury 


aJ = 0.723 # 5.2
EccJ = 0.0068 # 0.049
peri_dJ = aJ-EccJ*aJ
aphe_dJ = aJ+EccJ*aJ
viy_calcJ = np.sqrt(((2*4*np.pi**2)*(1/peri_dJ - 1/aphe_dJ))/(1-(peri_dJ/aphe_dJ)**2))

Ms = 1.0            # Mass of the sun in solar mass units 
Mm = 0.16601*10**(-6)
Mj = 2.4478383*10**(-6)  #954.79194*10**(-6)
G = 4*np.pi**2      # Gravitational constant G 

steps = 500     # Number of steps plotted over, an increase in the number of steps makes the orbit more and more correct, and the orbits begins to overlap with itself
dt = T/steps        # Length of a step defined by the period and how many steps there are 
t = np.arange(0.0, 3*T, dt)   #array of each step up to the period

initial_position_merc = [xi, yi]
inital_position_jup =  [-aJ, 0.0]
initial_position_sun = [0, 0]

initial_velocity_merc = [vix, viy]
initial_velocity_jup = [0, viy_calcJ]
initial_velocity_sun = [0, 0]

initial_conditions_new = np.array([
    initial_position_merc, inital_position_jup,
    initial_velocity_merc, initial_velocity_jup
    ]).ravel()


# Define the system of ODEs
def system_of_odes(t, S, m1, m2):
    m3 = 1.0
    p3 = [0, 0]
    # p1, p2, p3, dp1_dt, dp2_dt, dp3_dt = S
    p1, p2 = S[0:2], S[2:4]
    dp1_dt, dp2_dt = S[4:6], S[6:8]

    f1, f2 = dp1_dt, dp2_dt

    df1_dt = (G*m3*(p3 - p1))/(np.linalg.norm(p3 - p1)**3) + (G*m2*(p2 - p1))/(np.linalg.norm(p2 - p1)**3)
    df2_dt = (G*m3*(p3 - p2))/(np.linalg.norm(p3 - p2)**3) + (G*m1*(p1 - p2))/(np.linalg.norm(p1 - p2)**3)

    return np.array([f1, f2, df1_dt, df2_dt]).ravel()


solution = solve_ivp(
    fun = system_of_odes,
    t_span = (0, 3*T),
    y0 = initial_conditions_new,
    t_eval = t,
    args = (Mm, Mj)
    )

t_sol = solution.t
mercx_sol = solution.y[0]
mercy_sol = solution.y[1]

jupx_sol = solution.y[2]
jupy_sol = solution.y[3]

sunx_sol = solution.y[4]
suny_sol = solution.y[5]

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

merc_plt, = ax.plot(mercx_sol, mercy_sol, 'green', label='Mercury', linewidth=1)
jup_plt, = ax.plot(jupx_sol, jupy_sol, 'red', label='Jupiter', linewidth=1)
# sun_plt, = ax.plot(sunx_sol, suny_sol, 'blue',label='Sun', linewidth=1)

merc_dot, = ax.plot([mercx_sol[-1]], [mercy_sol[-1]], 'o', color='green', markersize=6)
jup_dot, = ax.plot([jupx_sol[-1]], [jupy_sol[-1]], 'o', color='red', markersize=6)
sun_dot, = ax.plot([0], [0], 'o', color='blue', markersize=6, label='Sun')


ax.set_title("The 3-Body Problem")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()

# def update(frame):
#     lower_lim = max(0, frame - 300)
#     print(f"Progress: {(frame+1)/len(t):.1%} | 100.0 %", end='\r')

#     x_current_1 = mercx_sol[lower_lim:frame+1]
#     y_current_1 = jupy_sol[lower_lim:frame+1]

#     x_current_2 = mercx_sol[lower_lim:frame+1]
#     y_current_2 = jupy_sol[lower_lim:frame+1]


#     merc_plt.set_data(x_current_1, y_current_1)  
#     merc_dot.set_data([x_current_1[-1]], [y_current_1[-1]])

#     jup_plt.set_data(x_current_2, y_current_2)  
#     jup_dot.set_data([x_current_2[-1]], [y_current_2[-1]])

#     return merc_plt, merc_dot, jup_plt, jup_dot

# animation = FuncAnimation(fig, update, frames=range(0, len(t), 2), interval=10, blit=True)
# plt.show()
