"""
Created on Sun Nov  3 16:13:51 2024

@author: oliver
"""

import matplotlib.pyplot as plt
import numpy as np
import EulerRK2_TEST as em

t = np.linspace(0, 2*np.pi, 10000)
u = em.xs + (em.a * em.Ecc)
v = em.ys

# Plot the orbit and the sun    
plt.figure()
plt.plot(em.EulerSoln[:, 0], em.EulerSoln[:, 1], label='Euler Method', color='Blue')        #Plot euler method
plt.plot(em.RK2Soln[:, 0], em.RK2Soln[:, 1], label="RK2 Orbit", color='Red')  #Plot RK2 method
plt.plot(em.RK4Soln[:, 0], em.RK4Soln[:, 1], label="RK4 Orbit", color='Green')  #Plot RK2 method
plt.scatter(em.xs, em.ys, marker='o', label='Sun', color = 'orange')            #Plot position of the sun


zeroes = np.linspace(0, 0, em.steps)
diam = np.linspace(u-em.a, u+em.a, em.steps)
plt.scatter(u, v, label='Theoretical Centre of Ellipse' , marker='o', color='red')  #Plot the centre of elipse considering sun at 0,0 and the semi major and eccintricity values
plt.plot(diam, zeroes)
plt.plot( u+em.a*np.cos(t) , v+em.b*np.sin(t), label='Calculated Method', color='orange')    #Plot the closed form soloution
plt.legend()
