

import mcts 
import numpy as np 
import matplotlib.pyplot as plt 

initial_state = np.array([
	[0,0,0,0],
	[1,0,0,0],
	])
n_timesteps = 40 
team_1_idxs = [0]
team_2_idxs = [1]
dt = 0.25

timesteps = [] 
states = [initial_state]
for timestep in range(n_timesteps):
	print(timestep/n_timesteps)
	_,team_1_action = mcts.best_action(states[timestep],True)
	_,team_2_action = mcts.best_action(states[timestep],False)

	old_state = states[timestep]
	new_state = np.zeros((old_state.shape))
	for idx in range(old_state.shape[0]):
		if idx in team_1_idxs:
			new_state[idx,:] = mcts.forward_per_robot(old_state[idx,:],team_1_action[idx,:])
		else:
			new_state[idx,:] = mcts.forward_per_robot(old_state[idx,:],team_2_action[idx,:])

	timesteps.append(timestep)
	states.append(new_state)

states = np.asarray(states) # print()
fig,ax = plt.subplots()
line = ax.plot(states[:,0,0],states[:,0,1],label='robot 0',linewidth=3)
ax.scatter(states[-1,0,0],states[-1,0,1],marker='o',color=line[0].get_color())
line = ax.plot(states[:,1,0],states[:,1,1],label='robot 1',linewidth=3)
ax.scatter(states[-1,1,0],states[-1,1,1],marker='o',color=line[0].get_color())
ax.scatter(0.6,0,label='goal',color='green',marker='o')
ax.grid(True)
ax.legend()
plt.show()

