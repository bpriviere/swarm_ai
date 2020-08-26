import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

data = np.loadtxt("output.csv", delimiter=',', ndmin=2,skiprows=1)
print(data.shape)


fig, axs = plt.subplots(2,2)
axs[0,0].axis('equal')

axs[0,0].plot(data[:,0], data[:,1], label="attacker")
# ax.plot(data[:,4], data[:,5], label="attacker")
# ax.plot(data[:,8], data[:,9], label="defender")
axs[0,0].plot(data[:,6], data[:,7], label="defender")

axs[0,0].set_xlim([0,0.5])
axs[0,0].set_ylim([0,0.5])
axs[0,0].legend()

# velocity
axs[1,0].plot(np.linalg.norm(data[:,2:4], axis=1))
axs[1,0].plot(np.linalg.norm(data[:,8:10], axis=1))

# acc
axs[1,1].plot(np.linalg.norm(data[:,4:5], axis=1))
axs[1,1].plot(np.linalg.norm(data[:,10:12], axis=1))

# reward
axs[0,1].plot(data[:,12])
axs[0,1].plot(data[:,13])

plt.show()