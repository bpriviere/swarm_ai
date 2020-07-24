

import mcts 
import numpy as np 
import matplotlib.pyplot as plt 

initial_state = np.array([
	[0,0,0,0],
	# [0,0.25,0,0],
	[0,1,0,0],
	[1,0,0,0],
	])
n_timesteps = 40 
goal = mcts.goal

timesteps = [] 
states = [initial_state]
for timestep in range(n_timesteps):

	print(timestep/n_timesteps)
	old_state = states[timestep]
	
	_,team_1_action = mcts.best_action(old_state,True)
	_,team_2_action = mcts.best_action(old_state,False)

	if team_1_action is not None and team_2_action is not None:
		new_state = np.zeros((old_state.shape))
		for idx in range(old_state.shape[0]):
			if idx in mcts.team_1_idxs:
				new_state[idx,:] = mcts.forward_per_robot(old_state[idx,:],team_1_action[idx,:])
			else:
				new_state[idx,:] = mcts.forward_per_robot(old_state[idx,:],team_2_action[idx,:])
		timesteps.append(timestep)
		states.append(new_state)

	else:
		break 

states = np.asarray(states) 
timesteps = np.asarray(timesteps)

fig,ax = plt.subplots()
ax.grid(True)
ax.scatter(mcts.goal[0],mcts.goal[1],label='goal',color='green',marker='o')
for i in range(initial_state.shape[0]):
	line = ax.plot(states[:,i,0],states[:,i,1],label='robot {}'.format(i),linewidth=3)
	ax.scatter(states[:,i,0],states[:,i,1],marker='o',color=line[0].get_color())
ax.legend()
# ax.set_aspect('equal')
plt.show()
