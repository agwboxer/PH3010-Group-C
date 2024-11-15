import numpy as np
import matplotlib.pyplot as plt

class GRMercuryOrbit:
    def __init__(self, G=39.478, M=1, c_prime=63.239, x0=0.307, y0=0, vx0=0, vy0=12.4375):
        self.G = G
        self.M = M
        self.c_prime = c_prime  # Modified speed of light
        self.initial_state = np.array([x0, y0, vx0, vy0])


    def derivatives(self, t, state, include_gr=True):
        """Calculate the derivatives for the state [x, y, vx, vy]."""
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        
        # Newtonian accelerations
        ax = -self.G * self.M * x / r**3
        ay = -self.G * self.M * y / r**3
        
        # GR correction term
        if include_gr:
            gr_correction = -3 * self.G**2 * self.M**2 / (2 * self.c_prime**2 * r**4)
            ax -= gr_correction * x
            ay -= gr_correction * y
        
        return np.array([vx, vy, ax, ay])
    
    def runge_kutta_4(self, t0, h, steps, include_gr=True):
        """Runge-Kutta 4th order method to integrate the equations of motion."""
        t = t0
        state = self.initial_state
        trajectory = [state[:2]]
        
        perihelions = []
        perihelion_angles = []
        previous_perihelion = float('inf')
        
        for i in range(steps):
            k1 = self.derivatives(t, state, include_gr)
            k2 = self.derivatives(t + h/2, state + h/2 * k1, include_gr)
            k3 = self.derivatives(t + h/2, state + h/2 * k2, include_gr)
            k4 = self.derivatives(t + h, state + h * k3, include_gr)
            
            state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            trajectory.append(state[:2])
            
            x, y = state[:2]
            r = np.sqrt(x**2 + y**2)
            
            # Track perihelion (minimum distance to the Sun)
            if r < previous_perihelion:
                previous_perihelion = r
            elif r > previous_perihelion:
                perihelions.append(previous_perihelion)
                
                perihelion_angle = np.arctan2(y, x)
                perihelion_angles.append(perihelion_angle)
                
                previous_perihelion = float('inf')
        
        # Calculate perihelion advance
        perihelion_advances = []
        for i in range(1, len(perihelion_angles)):
            delta_angle = perihelion_angles[i] - perihelion_angles[i - 1]
            
            if delta_angle < 0:
                delta_angle += 2 * np.pi
            
            perihelion_advances.append(delta_angle)
        
        return np.array(trajectory), perihelions, perihelion_advances
