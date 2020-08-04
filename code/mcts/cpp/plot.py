import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

data = np.loadtxt("output.csv", delimiter=',')


fig, ax = plt.subplots()
ax.axis('equal')

ax.plot(data[:,0], data[:,1], label="attacker")
ax.plot(data[:,4], data[:,5], label="defender")

ax.set_xlim([0,0.5])
ax.set_ylim([0,0.5])

ax.legend()
plt.show()