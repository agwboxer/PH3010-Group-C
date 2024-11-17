from shiny.express import input, ui
from shiny import render, reactive
import numpy as np
import matplotlib.pyplot as plt
from shinyswatch import theme

from GR_corr import GRMercuryOrbit
from solver import MercuryOrbit
from MethodComparison import OrbitComparison

# UI Configuration
ui.page_opts(title="Numerical Methods of Modelling the Orbit of Mercury", theme=theme.quartz)

with ui.sidebar():
    ui.input_select("select", "Select type", 
                    choices=["Method Comparison", "Newtonian", "General Relativity Correction"])

@render.ui
def simulation_ui():
    """Render additional UI components based on selected simulation type."""
    if input.select() == "Method Comparison":
        return ui.TagList(
            ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
            ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
            ui.input_slider("steps", "Max Number of Steps", min=10, max=2000, value=50, step=10)
        )
    elif input.select() == "Newtonian":
        return ui.TagList(
            ui.input_slider("steps", "Time Steps", min=50, max=25000, value=100, step=100),
            ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
            ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01)
        )
    return None

@reactive.Calc
def get_orbit():
    """Get the orbit instance based on the selected option."""
    if input.select() == "General Relativity Correction":
        return GRMercuryOrbit()
    elif input.select() == "Newtonian":
        ecc = input.eccentricity()
        a = input.a()
        steps = input.steps()
        return MercuryOrbit(ecc, a, steps)
    elif input.select() == "Method Comparison":
        ecc = input.eccentricity()
        a = input.a()
        return OrbitComparison(ecc, a)
    return None

def plotter(orbit):
    """Plot the orbit based on the selected option."""
    if isinstance(orbit, MercuryOrbit):
        # Plot Newtonian orbits using Euler, RK2, and RK4 methods
        t_euler, sol_euler = orbit.euler()
        t_rk2, sol_rk2 = orbit.rk2()
        t_rk4, sol_rk4 = orbit.rk4()

        plt.figure(facecolor='lightblue')
        plt.plot(sol_euler[:, 0], sol_euler[:, 1], label='Euler Method', color='red')
        plt.plot(sol_rk2[:, 0], sol_rk2[:, 1], label='RK2 Method', color='orange')
        plt.plot(sol_rk4[:, 0], sol_rk4[:, 1], label='RK4 Method', color='black')
        plt.scatter(0, 0, color='yellow', label='Sun')
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Orbit of Mercury")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.gca().set_facecolor('tab:blue')
    
    elif isinstance(orbit, GRMercuryOrbit):
        # Plot orbit with GR correction using RK4 method
        trajectory, _, perihelion = orbit.runge_kutta_4(0, 1/365, 3650)
        perihelion_avg = np.mean(perihelion)

        @render.text
        def text():
            return f"Perihelion advance per revolution: {perihelion_avg:.6f} arcseconds"

        plt.figure(facecolor='lightblue')
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='GR Correction', color='red')
        plt.axis('equal')
        plt.scatter(0, 0, color='yellow', label='Sun')
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Orbit of Mercury")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.gca().set_facecolor('tab:blue')

    elif isinstance(orbit, OrbitComparison):
        steps_data = np.linspace(10, input.steps(), 50)
        store_t_data, store_E_dist, store_RK2_dist, store_RK4_dist = orbit.run_simulation(steps_data)
        plt.figure(facecolor='lightblue')
        plt.scatter(store_t_data, store_E_dist, label='Euler', marker='.', color = 'red')
        plt.scatter(store_t_data, store_RK2_dist, label='RK2', marker='.', color = 'yellow')
        plt.scatter(store_t_data, store_RK4_dist, label='RK4', marker='.', color = 'orange')
        plt.xlabel('No. of timesteps')
        plt.ylabel('Distance from start (AU)')
        plt.title('Effect of timestep on orbital accuracy')
        plt.legend()
        plt.grid()
        plt.gca().set_facecolor('tab:blue')

    return plt.gcf()

@render.plot
def plot_orbit():
    """Render the plot based on the selected simulation type."""
    orbit = get_orbit()
    if orbit:
        return plotter(orbit)
