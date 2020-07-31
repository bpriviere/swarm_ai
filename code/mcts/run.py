

import mcts
import numpy as np 
import matplotlib.pyplot as plt 
import multiprocessing as mp
import glob
import os 
import shutil 

import sys
sys.path.append("../")
from param import Param 
from env import Swarm
import datahandler
import plotter 

def format_dir(param):

	# make current results dir 
	if not os.path.exists(param.current_results_dir + '/*'):
		os.makedirs(param.current_results_dir + '/*',exist_ok=True)

	# clean current results dir 
	for old_sim_result_dir in glob.glob(param.current_results_dir):
		shutil.rmtree(old_sim_result_dir)


def run_sim(param):

	# prep (just to get initial state)
	env = Swarm(param)
	reset = env.get_reset()
	state_vec = reset["state_initial"] 

	state = np.reshape(state_vec,(param.num_nodes,4))
	done = [] 
	turn = True

	tree = mcts.Tree(param)
	state = mcts.State(state,done,turn)

	# run sim 
	times,actions,dones,states,values = [],[],[state.done],[state.state],[]
	for step,time in enumerate(param.sim_times):

		print('\t\t t = {}/{}'.format(step,len(param.sim_times)))
		save_action = np.zeros((param.num_nodes,2))
		for team in range(2):

			tree = mcts.Tree(param)
			tree.set_root(state) 
			tree.grow()
			state, action = tree.best_action()
			
			if state is None: 
				break 

			save_action += action 
		
		if state is None: 
			break 

		values.append((\
			tree.root_node.value_1/tree.root_node.number_of_visits,\
			tree.root_node.value_2/tree.root_node.number_of_visits))
		times.append(time) 
		dones.append(state.done) 
		states.append(state.state) 
		actions.append(save_action)


	#  
	sim_result = dict()
	sim_result["times"] = np.asarray(times)
	sim_result["actions"] = np.asarray(actions) 
	sim_result["dones"] = np.asarray(dones[1:])
	sim_result["states"] = np.asarray(states[1:]) 
	sim_result["values"] = np.asarray(values) 
	sim_result["param"] = param.to_dict() 	

	# write sim results
	print('writing sim result...')
	case = len(glob.glob(param.current_results_dir + '/*'))
	results_dir = param.current_results_dir + '/sim_result_{}'.format(case)
	datahandler.write_sim_result(sim_result,results_dir)


if __name__ == '__main__':

	param = Param()
	param.current_results_dir = '../'+param.current_results_dir
	format_dir(param)

	parallel = True
	if parallel: 
		ncases = 50
		nprocess = np.min((mp.cpu_count()-1,ncases))
		pool = mp.Pool(nprocess)
		for _ in pool.imap_unordered(run_sim, [param for _ in range(nprocess)]):
			pass 
	else: 
		run_sim(param)

	sim_results = [] 
	for sim_result_dir in glob.glob(param.current_results_dir + '/*'):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result)

	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")