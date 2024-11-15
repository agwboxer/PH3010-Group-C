from shiny.express import input, ui
from shiny import render, reactive
import numpy as np
import matplotlib.pyplot as plt
from GR_corr import GRMercuryOrbit
from solver import MercuryOrbit

ui.page_opts(title="Numerical methods of Modelling the orbit of Mercury")


with ui.sidebar():
    ui.input_select("select", "Select type", choices=["Newtonian", "General Relativity Correction"])

@render.ui
def simulation_ui():
    """Render additional UI components if 'Newtonian' is selected."""
    if input.select() == "Newtonian":
        return ui.TagList(
            ui.input_slider("eccentricity", "Eccentricity", min=0.0, max=0.9, value=0.2056, step=0.01),
            ui.input_slider("a", "Semi-major Axis (AU)", min=0.1, max=1.0, value=0.387, step=0.01),
            ui.input_slider("steps", "Time Steps", min=50, max=25000, value=100, step=100)
        )
    return None

@reactive.Calc
def get_orbit():
    """Get the orbit instance based on the selected option."""
    if input.select() == "General Relativity Correction":
        # Return an instance of the GRMercuryOrbit class with GRplot enabled
        return GRMercuryOrbit()
    elif input.select() == "Newtonian":
        # Return the MercuryOrbit instance for Newtonian calculations
        ecc = input.eccentricity()
        a = input.a()
        steps = input.steps()
        return MercuryOrbit(ecc, a, steps)
    return None

def plot_combined_orbit(orbit):
    """Plot the orbit based on the selected option."""
    plt.figure()

    if isinstance(orbit, MercuryOrbit):
        # Plot Newtonian orbits using Euler, RK2, and RK4 methods
        t_euler, sol_euler = orbit.euler()
        t_rk2, sol_rk2 = orbit.rk2()
        t_rk4, sol_rk4 = orbit.rk4()
        
        plt.plot(sol_euler[:, 0], sol_euler[:, 1], label='Euler Method')
        plt.plot(sol_rk2[:, 0], sol_rk2[:, 1], label='RK2 Method')
        plt.plot(sol_rk4[:, 0], sol_rk4[:, 1], label='RK4 Method')
    
    elif isinstance(orbit, GRMercuryOrbit):
        # Plot the orbit with GR correction using RK4 method
        trajectory, _, _ = orbit.runge_kutta_4(0, 1/365, 3650)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='GR Correction')
        plt.axis('equal')

    plt.scatter(0, 0, color='yellow', label='Sun')
    plt.xlabel("x (AU)")
    plt.ylabel("y (AU)")
    plt.title("Orbit of Mercury")
    plt.grid(True)
    plt.legend(loc='upper left')
    return plt.gcf()


@render.plot
def plot_orbit():
    """Plot based on the selected option."""
    orbit = get_orbit()
    if orbit:
        return plot_combined_orbit(orbit)

#from shared import app_dir

#ui.include_css(app_dir / "styles.css")
