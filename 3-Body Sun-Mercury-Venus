import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 39.47841760435743  # Gravitational constant in AU^3 / (M_sun * day^2)
M_sun = 1.0  # Mass of the Sun in Solar masses
M_mercury = 1.652e-7  # Mass of Mercury in Solar masses
M_venus = 2.447e-6    # Mass of Venus in Solar masses

# Orbital radii in AU
r_mercury = 0.387
r_venus = 0.723

#  initial tangential velocities (circular approximation)
v_mercury = np.sqrt(G * M_sun / r_mercury)  # in AU/day
v_venus = np.sqrt(G * M_sun / r_venus)  # in AU/day

# Initial conditions (AU, AU/day)
initial_conditions = [
    0, 0, 0, 0, 0, 0,                # Sun: stationary at the origin
    r_mercury, 0, 0, 0, v_mercury, 0,  # Mercury: positioned along X-axis, moving along Y
    r_venus, 0, 0, 0, v_venus, 0      # Venus: positioned along X-axis, moving along Y
]

def gravitational_acceleration(mass, pos1, pos2):
    """the gravitational acceleration due to gravity between two bodies."""
    r_vec = pos2 - pos1
    r_mag = np.linalg.norm(r_vec)
    if r_mag == 0:
        return np.zeros(3)
    return G * mass * r_vec / r_mag**3

def three_body_dynamics(t, state):
    """ the derivatives for the 3-body system in AU and AU/day."""
    #  positions and velocities
    sun_pos = state[0:3]
    mercury_pos = state[6:9]
    venus_pos = state[12:15]
    
    sun_vel = state[3:6]
    mercury_vel = state[9:12]
    venus_vel = state[15:18]

    #  accelerations
    mercury_acc = gravitational_acceleration(M_sun, mercury_pos, sun_pos) + \
                  gravitational_acceleration(M_venus, mercury_pos, venus_pos)
    
    venus_acc = gravitational_acceleration(M_sun, venus_pos, sun_pos) + \
                gravitational_acceleration(M_mercury, venus_pos, mercury_pos)

    # Sun remains stationary (approximation)
    sun_acc = np.zeros(3)
    
    #  derivatives for ODE solver
    derivatives = np.concatenate((sun_vel, sun_acc, mercury_vel, mercury_acc, venus_vel, venus_acc))
    return derivatives


t_start = 0
t_end = 365  
num_steps = 100000  

# Solving the system with less strict tolerances for faster performance
sol = solve_ivp(three_body_dynamics, (t_start, t_end), initial_conditions, 
                t_eval=np.linspace(t_start, t_end, num_steps), 
                rtol=1e-6, atol=1e-6, method='LSODA')

# Extract positions for plot
sun_pos = sol.y[0:3, :]
mercury_pos = sol.y[6:9, :]
venus_pos = sol.y[12:15, :]

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adjusting the axis limits to give enough space for the orbits
ax.set_xlim(-1.5, 1.5)  # X-axis limit
ax.set_ylim(-1.5, 1.5)  # Y-axis limit
ax.set_zlim(-0.1, 0.1)      # Z-axis limit 

# Plotting the orbits of Mercury and Venus
ax.plot(mercury_pos[0], mercury_pos[1], mercury_pos[2], '-', color="orange", label="Mercury", markersize=0.1)
ax.plot(venus_pos[0], venus_pos[1], venus_pos[2], '-', color="green", label="Venus", markersize=0.1)

# Plot the Sun at the origin
ax.plot([0], [0], [0], 'yo', markersize=10, label="Sun")

# Set the viewing angle for a better 3D perspective
ax.view_init(elev=30, azim=60)

# Labels and legend
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_zlabel('Z (AU)')
ax.legend()

# Displaying the plot
plt.show()
