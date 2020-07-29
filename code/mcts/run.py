

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

	#  
	sim_result = dict()
	sim_result["times"] = np.asarray(times)
	sim_result["dones"] = np.asarray(dones)
	sim_result["states"] = np.asarray(states) 
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

	parallel = False 
	if parallel: 
		ncases = 5
		nprocess = np.min((mp.cpu_count(),ncases))		
		pool = mp.Pool(nprocess)
		for _ in pool.imap_unordered(run_sim, [param for _ in range(nprocess)]):
			pass 
	else: 
		run_sim(param)

	# load sim results 
	sim_results = [] # lst of dicts
	for sim_result_dir in glob.glob(param.current_results_dir + '/*'):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	# plots 
	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result)

	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")