import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PlanetarySystem:
    def __init__(self, factor=30, steps=750):
        # Constants and initial conditions
        self.G = 4 * np.pi ** 2  # Gravitational constant
        self.c_prime = 63.239    # Speed of light correction factor for GR
        self.factor = factor
        self.steps = steps
        
        # Sun's mass and positions
        self.Ms = 1.0
        self.ps = np.array([0, 0])
        
        # Mercury parameters
        self.Ecc, self.a = 0.2056, 0.387
        self.b = self.a * np.sqrt(1 - self.Ecc ** 2)
        self.peri_d, self.aphe_d = self.a - self.Ecc * self.a, self.a + self.Ecc * self.a
        self.viy_calc = np.sqrt(((2 * 4 * np.pi ** 2) * (1 / self.peri_d - 1 / self.aphe_d)) / (1 - (self.peri_d / self.aphe_d) ** 2))
        self.merc_pos = [-self.peri_d, 0]
        self.merc_vel = [0, self.viy_calc]

        # Venus parameters
        self.EccV, self.aV = 0.0068, 0.723
        self.peri_dV, self.aphe_dV = self.aV - self.EccV * self.aV, self.aV + self.EccV * self.aV
        self.viy_calcV = np.sqrt(((2 * 4 * np.pi ** 2) * (1 / self.peri_dV - 1 / self.aphe_dV)) / (1 - (self.peri_dV / self.aphe_dV) ** 2))
        self.ven_pos = [-self.peri_dV, 0]
        self.ven_vel = [0, self.viy_calcV]

        # Mass of planets (in solar mass units)
        Ms_kg = 1.988416e30
        self.Mm = 0.33010e24 / Ms_kg
        self.Mv = 4.8673e24 / Ms_kg

        # Time settings
        Tsq = self.a ** 3
        self.T = np.sqrt(Tsq)
        self.dt = self.T / self.steps
        self.t = np.arange(0.0, self.factor * self.T, self.dt)

        # Initial conditions
        self.initial_conditions = np.array([
            *self.merc_pos, *self.ven_pos, *self.merc_vel, *self.ven_vel
        ])

    def system_of_odes(self, t, S, m1, m2):
        p1, p2 = S[0:2], S[2:4]
        dp1_dt, dp2_dt = S[4:6], S[6:8]

        df1_dt = ((self.G * self.Ms * (self.ps - p1)) / (np.linalg.norm(self.ps - p1) ** 3) +
                  (self.G * m2 * (p2 - p1)) / (np.linalg.norm(p2 - p1) ** 3))
        df2_dt = ((self.G * self.Ms * (self.ps - p2)) / (np.linalg.norm(self.ps - p2) ** 3) +
                  (self.G * m1 * (p1 - p2)) / (np.linalg.norm(p1 - p2) ** 3))

        return np.array([dp1_dt, dp2_dt, df1_dt, df2_dt]).ravel()

    def solve_orbits(self, use_gr=False):
        # Choose the appropriate ODE system
        system_func = self.system_of_odes
        
        solution = solve_ivp(
            fun=system_func,
            t_span=(0, self.factor * self.T),
            y0=self.initial_conditions,
            t_eval=self.t,
            args=(self.Mm, self.Mv),
            method='RK23'
        )
        return solution

    def plot_orbits(self, solution):
        mercx_sol, mercy_sol = solution.y[0], solution.y[1]
        venx_sol, veny_sol = solution.y[2], solution.y[3]

        fig, ax = plt.subplots()
        ax.plot(mercx_sol, mercy_sol, 'green', label='Mercury', linewidth=1)
        ax.plot(venx_sol, veny_sol, 'red', label='Venus', linewidth=1)
        ax.plot([0], [0], 'o', color='yellow', markersize=6, label='Sun')

        ax.set_title("2 Planets Orbit Around the Sun")
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend()
        plt.show()

    def animate_orbits(self, solution):
        mercx_sol, mercy_sol = solution.y[0], solution.y[1]
        venx_sol, veny_sol = solution.y[2], solution.y[3]

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        merc_plt, = ax.plot([], [], [], 'green', linewidth=1, label='Mercury')
        ven_plt, = ax.plot([], [], [], 'red', linewidth=1, label='Venus')
        sun_dot, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=6)

        def update(frame):
            lower_lim = max(0, frame - 300)
            x1, y1 = mercx_sol[:frame+1], mercy_sol[:frame+1]
            x2, y2 = venx_sol[:frame+1], veny_sol[:frame+1]

            merc_plt.set_data_3d(x1, y1, [0] * len(x1))
            ven_plt.set_data_3d(x2, y2, [0] * len(x2))
            return merc_plt, ven_plt

        ani = FuncAnimation(fig, update, frames=range(0, len(self.t), 5), interval=5, blit=True)
        plt.show()


if __name__ == "__main__":
    # Instantiate the class and solve the system
    planetary_system = PlanetarySystem()
    solution = planetary_system.solve_orbits()
    
    # Plot or animate the solution
    planetary_system.plot_orbits(solution)
    planetary_system.animate_orbits(solution)
