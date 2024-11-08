import matplotlib.pyplot as plt
import numpy as np
import EulerRK2_TEST as em

t = np.linspace(0, 2*np.pi, em.steps)
u = em.xs + (em.a * em.Ecc)
v = em.ys
x_theory = u+em.a*np.cos(t)
y_theory =v+em.b*np.sin(t)

# Plot the orbit and the sun    
plt.figure()
plt.plot(em.EulerSoln[:, 0], em.EulerSoln[:, 1], label='Euler Method', color='Blue')      #Plot euler method
plt.plot(em.RK2Soln[:, 0], em.RK2Soln[:, 1], label="RK2 Orbit", color='Red')              #Plot RK2 method
plt.plot(em.RK4Soln[:, 0], em.RK4Soln[:, 1], label="RK4 Orbit", color='Green')            #Plot RK4 method
plt.scatter(em.xs, em.ys, marker='o', label='Sun', color = 'orange')                      #Plot position of the sun


#Additional plotting that can be done to verify results
zeroes = np.linspace(0, 0, em.steps)
diam = np.linspace(u-em.a, u+em.a, em.steps)
plt.scatter(u, v, label='Theoretical Centre of Ellipse' , marker='o', color='red')          #Plot the centre of elipse considering sun at 0,0 and the semi major and eccintricity values
plt.plot(diam, zeroes)
plt.plot(x_theory , y_theory, label='Calculated Method', color='orange')    #Plot the closed form soloution based on theoretical equations
plt.legend()
plt.title(f'Steps = {em.steps},  $\delta$t = {em.dt} years')
