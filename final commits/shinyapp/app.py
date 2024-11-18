from shiny.express import input, ui
from shiny import render, reactive
import numpy as np
import matplotlib.pyplot as plt
from shinyswatch import theme

from GR_Corrections import GR_Orbit
from Newtonian_Simulation import Newtonian_Orbit
from MethodComparison import Numerical_Comparison
from Three_Body_Simulation import ThreeBodySimulation

# UI Configuration, sets title and page theme
ui.page_opts(title="Numerical Methods of Modelling the Orbit of Mercury", theme=theme.morph)

# Create UI side bar for selecting simulation type
with ui.sidebar():
    ui.input_select("select", "Select type", 
                    choices=["Method Comparison", "Newtonian", "General Relativity Correction", "Three Body"])

@render.ui
def simulation_ui():
    """
    Render additional UI components based on the selected simulation type

    Returns:
    - UI elements: Depending on the selected simulation type, relevant UI inputs are displayed
    """
    if input.select() == "Method Comparison":
        return ui.TagList(
            ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
            ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
            ui.input_slider("steps", "Max Number of Steps", min=10, max=2000, value=50, step=10),
            ui.input_switch("Curvefit", "Toggle curve fit graph", value=False)
        )
    elif input.select() == "Newtonian":
        return ui.TagList(
            ui.input_slider("steps", "Time Steps", min=50, max=25000, value=100, step=100),
            ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
            ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
        )
    elif input.select() == "Three Body":
        return ui.TagList(
            ui.input_slider("steps", "Time Steps", min=1000, max = 250000, value=100000, step=1000),
            ui.input_slider("t_end", "Simulation time (Days)", min = 1, max = 36500, value = 365, step = 50)
        )
    return None


@reactive.Calc
def get_orbit():
    """
    Get the appropriate orbit instance based on the selected simulation type and initialise class with values selected from input sliders

    Returns:
    - Object: The corresponding orbit simulation object based on the user input
    """
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
        return MercuryOrbitSimulation(ecc, a)
    elif input.select() == "Three Body":
        steps = input.steps()
        t_end = input.t_end()
        return ThreeBodySimulation(0, t_end, steps)
    return None

def plotter(orbit):
    """
    Plot the orbit based on the selected simulation type

    Parameters:
    - orbit (object): The orbit object to plot (either Newtonian, GR correction, Method Comparison, or Three Body)

    Returns:
    - matplotlib axis: The axis on which the plot is drawn
    """
    if isinstance(orbit, MercuryOrbit):
        # Plot Newtonian orbits using Euler, RK2, and RK4 methods
        _, sol_euler = orbit.euler()
        _, sol_rk2 = orbit.rk2()
        _, sol_rk4 = orbit.rk4()

        plt.figure()
        plt.plot(sol_euler[:, 0], sol_euler[:, 1], label='Euler Method', color='red')
        plt.plot(sol_rk2[:, 0], sol_rk2[:, 1], label='RK2 Method', color='orange')
        plt.plot(sol_rk4[:, 0], sol_rk4[:, 1], label='RK4 Method', color='black')
        plt.scatter(0, 0, color='yellow', label='Sun')
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Orbit of Mercury")
        plt.grid(True)
        plt.legend(loc='upper left', frameon=False)

    elif isinstance(orbit, GRMercuryOrbit):
        # Plot orbit with GR correction using RK4 method
        trajectory, _, perihelion = orbit.runge_kutta_4(0, 1/365, 3650)
        perihelion_avg = np.mean(perihelion)

        # Displays text about the perihelion advance
        @render.text
        def text():
            return f"Perihelion advance per revolution: {perihelion_avg} arcseconds"

        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='GR Correction', color='red')
        plt.axis('equal')
        plt.scatter(0, 0, color='yellow', label='Sun')
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Orbit of Mercury")
        plt.grid(True)
        plt.legend(loc='upper left', frameon=False)

    elif isinstance(orbit, MercuryOrbitSimulation):
        steps_data = np.linspace(10, input.steps(), 50)
        store_t_data, store_E_dist, store_RK2_dist, store_RK4_dist = orbit.run_simulation(steps_data)

        # Plot 1: Effect of timestep on orbital accuracy
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)  # First subplot
        plt.plot(store_t_data, store_E_dist, label='Euler', marker='.', color='red')
        plt.plot(store_t_data, store_RK2_dist, label='RK2', marker='.', color='yellow')
        plt.plot(store_t_data, store_RK4_dist, label='RK4', marker='.', color='orange')
        plt.xlabel('No. of timesteps')
        plt.ylabel('Distance from start (AU)')
        plt.title('Effect of Timestep on Orbital Accuracy')
        plt.legend(frameon=False)
        plt.grid()

        # Plot 2: Convergence plot (Error vs Step Size)
        if input.Curvefit():
            global_error_euler, global_error_rk2, global_error_rk4, step_sizes = orbit.compute_errors(steps_data)

            plt.subplot(2, 1, 2)  # Second subplot
            plt.loglog(step_sizes, global_error_euler, marker=".", label='Euler')
            plt.loglog(step_sizes, global_error_rk2, marker=".", label='RK2')
            plt.loglog(step_sizes, global_error_rk4, marker=".", label='RK4')
            plt.xlabel('Step Size')
            plt.ylabel('Global Error (AU)')
            plt.title('Convergence Plot')
            plt.legend(frameon=False)
            plt.grid()

    elif isinstance(orbit, ThreeBodySimulation):
        # Run three body simulation and gather valuess
        sol = orbit.run_simulation()
        mercury_pos = sol.y[6:9, :]
        venus_pos = sol.y[12:15, :]

        # Set up the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Adjusting the axis limits to give enough space for the orbits
        ax.set_xlim(-1.5, 1.5)  
        ax.set_ylim(-1.5, 1.5)  
        ax.set_zlim(-0.1, 0.1)  

        # Plotting the orbits of Mercury and Venus
        ax.plot(mercury_pos[0], mercury_pos[1], mercury_pos[2], '-', color="orange", label="Mercury", markersize=0.1)
        ax.plot(venus_pos[0], venus_pos[1], venus_pos[2], '-', color="green", label="Venus", markersize=0.1)

        # Plot the Sun at the origin
        ax.plot([0], [0], [0], 'yo', markersize=10, label="Sun")

        # Set the viewing angle for a better 3D perspective
        ax.view_init(elev=30, azim=60)

        # Adjust labels and remove frame from legend to make it clearer
        ax.set_xlabel('X (AU)')
        ax.set_ylabel('Y (AU)')
        ax.set_zlabel('Z (AU)')
        ax.legend(frameon=False)

    return plt.gca()

@render.plot
def plot_orbit():
    """
    Render the plot based on the selected simulation type.

    Returns:
    - matplotlib axis: The plot is rendered on this axis.
    """
    orbit = get_orbit()
    if orbit:
        return plotter(orbit)
