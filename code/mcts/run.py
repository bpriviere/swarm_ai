

import mcts 
import numpy as np 
import matplotlib.pyplot as plt 

import sys
sys.path.append("../")
from param import Param 
from env import Swarm

def main():

	# prep (just to get initial state)
	param = Param()
	env = Swarm(param)
	reset = env.get_reset()
	state_vec = reset["state_initial"] 

	robots_state = np.reshape(state_vec,(param.num_nodes,4))
	robots_done = np.zeros(param.num_nodes_A)
	team_1_turn = True

	tree = mcts.Tree(param)
	state = mcts.State(param,robots_state,robots_done,team_1_turn)

	# run sim 
	times,dones,states = [],[state.robots_done],[state.robots_state]
	for step,time in enumerate(param.sim_times):

		print('\t\t t = {}/{}'.format(step,len(param.sim_times)))
		for team in range(2):

			adaptive_on = False
			if adaptive_on:
				tree.set_root(state) 
				tree.grow()

			else:
				tree = mcts.Tree(param)
				tree.set_root(state) 
				tree.grow()

			state, action = tree.best_action()

			times.append(time)
			dones.append(state.robots_done)
			states.append(state.robots_state)

		if np.all(state.robots_done):
			break 

	# plot 
	times = np.asarray(times)
	dones = np.asarray(dones)
	states = np.asarray(states) 

	team_1_color = 'blue'
	team_2_color = 'orange'
	goal_color = 'green'

	fig,ax = plt.subplots()
	ax.grid(True)
	ax.scatter(param.goal[0],param.goal[1],label='goal',color=goal_color,marker='o')
	ax.plot(np.nan,np.nan,color=team_1_color,label='attackers')
	ax.plot(np.nan,np.nan,color=team_2_color,label='defenders')
	for i in range(param.num_nodes):
		color = team_2_color 
		if i in param.team_1_idxs:
			color = team_1_color
		ax.plot(states[:,i,0],states[:,i,1],linewidth=3,color=color)
		ax.scatter(states[:,i,0],states[:,i,1],marker='o',color=color)
	ax.legend()
	plt.show()

if __name__ == '__main__':
	main()


# class Param:
# 	def __init__(self):

# 		# tree param 
# 		self.c_param = 1.4 
# 		self.num_simulations = 10
# 		self.rollout_horizon = 10 
# 		self.num_nodes = 10

# 		# sim param 
# 		self.initial_robot_state = np.array([
# 			[0,0,0,0],
# 			[0,0.25,0,0],
# 			[0,1,0,0],
# 			[1,0,0,0],
# 			])
# 		self.n_timesteps = 40 

# 		# parameters
# 		self.r_capture = 0.05 
# 		self.goal = np.array([0.45,0.05]) 
# 		self.u_max_1 = 0.1 
# 		self.u_max_2 = 0.2 
# 		self.v_max_1 = 0.1 
# 		self.v_max_2 = 0.2 
# 		self.dt 		= 0.25 

# 		self.num_nodes_A = 1 
# 		self.num_nodes_B = 1 

# 	def update(self):

# 		self.team_1_idxs = []
# 		self.team_2_idxs = [] 

# 		for i in range(self.num_nodes_A + self.num_nodes_B):
