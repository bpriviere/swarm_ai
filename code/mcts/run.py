

import mcts2 as mcts
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

	state = np.reshape(state_vec,(param.num_nodes,4))
	done = [] 
	turn = True

	tree = mcts.Tree(param)
	state = mcts.State(param,state,done,turn)

	# run sim 
	times,dones,states = [],[state.done],[state.state]
	for step,time in enumerate(param.sim_times):

		print('\t\t t = {}/{}'.format(step,len(param.sim_times)))
		for team in range(2):

			tree = mcts.Tree(param)
			tree.set_root(state) 
			tree.grow()
			state, action = tree.best_action()

		if len(state.done) == len(param.team_1_idxs):
			break 

		times.append(time)
		dones.append(state.done)
		states.append(state.state)

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
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	plt.show()

if __name__ == '__main__':
	main()
