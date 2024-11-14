from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import numpy as np
from solver import MercuryOrbit
from GR_corr import GRMercuryOrbit

# Define the UI
app_ui = ui.page_fluid(
    ui.input_select(
        "select",
        "Select an option below:",
        {"GR": "General Relativity Correction", "NW": "Newtonian"},
    ),
    ui.output_text_verbatim("value"),
    ui.output_ui("simulation_ui")
)

def server(input, output, session):
    @output
    @render.text
    def value():
        """Render text based on the dropdown selection."""
        if input.select() == "GR":
            return "General Relativity Correction selected"
        elif input.select() == "NW":
            return "Newtonian selected"
        else:
            return "Select an option"

    @output
    @render.ui
    def simulation_ui():
        """Render additional UI components if 'Newtonian' is selected."""
        if input.select() == "NW":
            return ui.TagList(
                ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
                ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
                ui.input_slider("steps", "Time Steps", min=50, max=25000, value=100, step=100),
                ui.output_plot("plot_euler"),
                ui.output_plot("plot_rk2"),
                ui.output_plot("plot_rk4")
            )
        return None

    @reactive.Calc
    def get_orbit():
        """Get the orbit instance based on the selected option."""
        if input.select() == "GR":
            return GRMercuryOrbit(GRplot=False)
        elif input.select() == "NW":
            ecc = input.eccentricity()
            a = input.a()
            steps = input.steps()
            return MercuryOrbit(ecc, a, steps)
        return None

    def plot_orbit(t, sol, title):
        """Helper function to plot the orbit."""
        plt.figure()
        plt.plot(sol[:, 0], sol[:, 1], label=title)
        plt.scatter(0, 0, color='yellow', label='Sun')
        plt.title(title)
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.grid(True)
        plt.legend(loc='upper left')
        return plt.gcf()
    
    def plot_GR_orbit(trajectory):
        """Plot the orbit of Mercury with GR correction."""
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Mercury orbit (with GR correction)')
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title('Orbit of Mercury around the Sun (GR Correction)')
        plt.grid()
        plt.legend()
        plt.axis('equal')
        return plt.gcf()

    @output
    @render.plot
    def plot_euler():
        orbit = get_orbit()
        if isinstance(orbit, MercuryOrbit):
            t, sol = orbit.euler()
            return plot_orbit(t, sol, "Euler Method")
        elif isinstance(orbit, GRMercuryOrbit):
            trajectory, _, _ = orbit.runge_kutta_4(0, 1/365, 3650, include_gr=True)
            return plot_GR_orbit(trajectory)

    @output
    @render.plot
    def plot_rk2():
        orbit = get_orbit()
        if isinstance(orbit, MercuryOrbit):
            t, sol = orbit.rk2()
            return plot_orbit(t, sol, "RK2 Method")
        elif isinstance(orbit, GRMercuryOrbit):
            trajectory, _, _ = orbit.runge_kutta_4(0, 1/365, 3650, include_gr=True)
            return plot_GR_orbit(trajectory)

    @output
    @render.plot
    def plot_rk4():
        orbit = get_orbit()
        if isinstance(orbit, MercuryOrbit):
            t, sol = orbit.rk4()
            return plot_orbit(t, sol, "RK4 Method")
        elif isinstance(orbit, GRMercuryOrbit):
            trajectory, _, _ = orbit.runge_kutta_4(0, 1/365, 3650, include_gr=True)
            return plot_GR_orbit(trajectory)

# Run the Shiny app
app = App(app_ui, server)
