import numpy as np
import matplotlib.pyplot as plt
from shiny import App, render, ui, reactive
from solver import PlanetOrbit  # Importing the PlanetOrbit class from solver

# Define the UI for the Shiny app
app_ui = ui.page_fluid(
    ui.h2("Orbit Simulation using Numerical Methods"),
    ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
    ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
    ui.input_slider("steps", "Time Steps", min=50, max=25000, value=100, step=100),
    ui.output_plot("plot_euler"),
    ui.output_plot("plot_rk2"),
    ui.output_plot("plot_rk4")
)

# Initialise shiny server and define functions to update plots based on input
def server(input, output, session):
    """
    Server function that handles user input and updates the plots dynamically.
    It reacts to changes in the input sliders and computes the orbital trajectories
    using different numerical methods (Euler, RK2, RK4).
    """
    @reactive.Effect
    def _():
        # Get the current input values from the sliders
        ecc = input.eccentricity()  # Eccentricity of the orbit
        a = input.a()  # Semi-major axis of the orbit (in AU)
        steps = input.steps()  # Number of time steps for the simulation

        # Initialize the orbit object with the input values
        orbit = PlanetOrbit(ecc=ecc, a=a, steps=steps)

        # Compute the orbital solutions using different numerical methods
        euler_t, euler_sol = orbit.euler()  # Euler method
        rk2_t, rk2_sol = orbit.rk2()  # Runge-Kutta 2nd order method
        rk4_t, rk4_sol = orbit.rk4()  # Runge-Kutta 4th order method

        # Plotting function for the orbits
        def plot_orbit(t, sol, title):
            """
            Helper function to plot the orbit based on the solution
            from numerical methods.
            """
            plt.figure()  # Create a new figure
            plt.plot(sol[:, 0], sol[:, 1], label=title)  # Plot the orbit path
            plt.scatter(0, 0, color='yellow', label='Sun')  # Mark the Sun at the origin
            plt.title(title)  # Set the title of the plot
            plt.xlabel("x (AU)")  # Label for the x-axis
            plt.ylabel("y (AU)")  # Label for the y-axis
            plt.grid(True)  # Add grid for better readability
            plt.legend()  # Show the legend to identify the Sun and the orbit method

        # Create the plots for each numerical method
        @output
        @render.plot
        def plot_euler():
            """
            Render the plot for the Euler method.
            """
            plot_orbit(euler_t, euler_sol, "Euler Method")

        @output
        @render.plot
        def plot_rk2():
            """
            Render the plot for the RK2 method.
            """
            plot_orbit(rk2_t, rk2_sol, "RK2 Method")

        @output
        @render.plot
        def plot_rk4():
            """
            Render the plot for the RK4 method.
            """
            plot_orbit(rk4_t, rk4_sol, "RK4 Method")


# Run the app
app = App(app_ui, server)
