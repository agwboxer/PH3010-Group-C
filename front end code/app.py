import numpy as np
import matplotlib.pyplot as plt
from shiny import render, App
from shiny.express import input, ui

# Load radii data from the simulation
radii = np.load("radii_data.npy")

# Define UI for the Shiny app
app_ui = ui.page_fluid(
    ui.input_slider("n", "Number of bins", min=5, max=100, value=20),
    ui.output_plot("hist_plot")
)

# Define server logic for the Shiny app
def server(input, output, session):
    @output
    @render.plot(alt="A histogram of orbital radii")
    def hist_plot():
        fig, ax = plt.subplots()
        ax.hist(radii, bins=input.n(), density=True)
        ax.set_title("Histogram of Orbital Radii")
        ax.set_xlabel("Radius (AU)")
        ax.set_ylabel("Density")
        return fig

# Run the app
app = App(app_ui, server)
