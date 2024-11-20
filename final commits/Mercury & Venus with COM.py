import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# all constants
G = 39.478  # gravitational constant
c = 63197  # in AU/year
mass_sun = 1.0  # in Solar masses
mass_mercury = 1.651e-7  # in Solar masses
mass_venus = 2.447e-6  # in Solar masses

# orbital details
semi_major_axis_mercury = 0.39  # in AU
semi_major_axis_venus = 0.72  # in AU
eccentricity_mercury = 0.205  # Mercury's orbital eccentricity
eccentricity_venus = 0.007  # Venus's orbital eccentricity

orbital_period_mercury = 0.2408  # in years
orbital_period_venus = 0.6152  # in years

# time span for simulation
t_max = orbital_period_venus  # one Venus year
t_eval = np.linspace(0, t_max, 1000)

# function to calculate initial velocity for elliptical orbits
def compute_initial_conditions(semi_major_axis, eccentricity, mass):
    periapsis = semi_major_axis * (1 - eccentricity)  # closest point to the barycenter
    # velocity at periapsis using the derived equation
    velocity_periapsis = np.sqrt(
        (2 * G * mass * (1 / periapsis - 1 / (semi_major_axis * (1 + eccentricity)))) /
        (1 - (periapsis / (semi_major_axis * (1 + eccentricity)))**2))

    return periapsis, velocity_periapsis

# initial conditions for Mercury
distance_mercury, velocity_mercury = compute_initial_conditions(semi_major_axis_mercury, eccentricity_mercury, mass_sun)
mercury_initial = [distance_mercury, 0, 0, velocity_mercury]

# initial conditions for Venus
distance_venus, velocity_venus = compute_initial_conditions(semi_major_axis_venus, eccentricity_venus, mass_sun)
venus_initial = [distance_venus, 0, 0, velocity_venus]

# define the ODE system with GR corrections
def system_of_odes_gr(t, state, G, M, c):
    """
    state contains positions and velocities in the format:
    [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)  # distance to the Sun

    # newtonian and GR-corrected acceleration
    accel_common = G * M / r**3 + 3 * G**2 * M**2 / (c**2 * r**4)
    ax = -accel_common * x
    ay = -accel_common * y

    return [vx, vy, ax, ay]

# solution for Mercury's orbit using RK45
solution_mercury = solve_ivp(
    system_of_odes_gr, (0, t_max), mercury_initial, args=(G, mass_sun, c), t_eval=t_eval, method='RK45'
)
x_mercury, y_mercury = solution_mercury.y[0], solution_mercury.y[1]

# solution for Venus's orbit
solution_venus = solve_ivp(
    system_of_odes_gr, (0, t_max), venus_initial, args=(G, mass_sun, c), t_eval=t_eval, method='RK45'
)
x_venus, y_venus = solution_venus.y[0], solution_venus.y[1]

# barycenter positions found at each time step
barycenter_x = (mass_mercury * x_mercury + mass_venus * x_venus) / (mass_mercury + mass_venus)
barycenter_y = (mass_mercury * y_mercury + mass_venus * y_venus) / (mass_mercury + mass_venus)

# visualisations
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)  # in AU
ax.set_ylim(-1.5, 1.5)  # in AU
ax.set_aspect('equal', 'box')
ax.set_title("Relativistic Orbits of Mercury and Venus with Barycenter")
ax.set_xlabel("X Position (AU)")
ax.set_ylabel("Y Position (AU)")

# Sun, Mercury, Venus, and barycenter points
sun_dot, = ax.plot(0, 0, 'yo', label="Sun", markersize=10)
mercury_dot, = ax.plot([], [], 'ro', label="Mercury", markersize=8)
venus_dot, = ax.plot([], [], 'bo', label="Venus", markersize=8)
barycenter_dot, = ax.plot([], [], 'go', label="Barycenter", markersize=6)

# orbit traces
mercury_trace, = ax.plot([], [], 'r-', linewidth=0.5, alpha=0.7)
venus_trace, = ax.plot([], [], 'b-', linewidth=0.5, alpha=0.7)
barycenter_trace, = ax.plot([], [], 'g-', linewidth=0.5, alpha=0.7)

# legend
ax.legend(loc="upper right")

# trace histories
mercury_history_x, mercury_history_y = [], []
venus_history_x, venus_history_y = [], []
barycenter_history_x, barycenter_history_y = [], []

def update(frame):
    mercury_history_x.append(x_mercury[frame])
    mercury_history_y.append(y_mercury[frame])
    venus_history_x.append(x_venus[frame])
    venus_history_y.append(y_venus[frame])
    barycenter_history_x.append(barycenter_x[frame])
    barycenter_history_y.append(barycenter_y[frame])

    # update positions
    mercury_dot.set_data(x_mercury[frame], y_mercury[frame])
    venus_dot.set_data(x_venus[frame], y_venus[frame])
    barycenter_dot.set_data(barycenter_x[frame], barycenter_y[frame])

    # update traces
    mercury_trace.set_data(mercury_history_x, mercury_history_y)
    venus_trace.set_data(venus_history_x, venus_history_y)
    barycenter_trace.set_data(barycenter_history_x, barycenter_history_y)
    return mercury_dot, venus_dot, barycenter_dot, mercury_trace, venus_trace, barycenter_trace

# create animation
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=30, blit=True)

# save as a GIF
ani.save("Orbits of Mercury and Venus.gif", writer='pillow', fps=30)

# display
plt.show()
