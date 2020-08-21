
import mcts
import numpy as np 
import matplotlib.pyplot as plt 
import multiprocessing as mp
import glob
import os 
import shutil 
import time as timer
import tempfile
import subprocess
import sys
import torch

sys.path.append("../")
from param import Param 
import datahandler as dh
import plotter 
from measurements.relative_state import relative_state
from learning.discrete_emptynet import DiscreteEmptyNet

sys.path.append('cpp')
from convertNN import convertNN
from buildRelease import mctscpp

sys.path.append("../glas/")
from grun import clean_files_containing_str_from_dir, config_to_game, state_to_game_state, value_to_dist 
from gparam import Gparam


def format_data(o_a,o_b,goal):
	# input: [num_a/b, dim_state_a/b] np array 
	# output: 1 x something torch float tensor

	# make 0th dim (this matches batch dim in training)
	if o_a.shape[0] == 0:
		o_a = np.expand_dims(o_a,axis=0)
	if o_b.shape[0] == 0:
		o_b = np.expand_dims(o_b,axis=0)
	goal = np.expand_dims(goal,axis=0)

	# reshape if more than one element in set
	if o_a.shape[0] > 1: 
		o_a = np.reshape(o_a,(1,np.size(o_a)))
	if o_b.shape[0] > 1: 
		o_b = np.reshape(o_b,(1,np.size(o_b)))

	o_a = torch.from_numpy(o_a).float() 
	o_b = torch.from_numpy(o_b).float()
	goal = torch.from_numpy(goal).float()

	return o_a,o_b,goal


def evaluate_stochastic_policy(param): 

	# init 
	generator = mctscpp.createRandomGenerator(param.seed)
	game = config_to_game(param,generator) 

	states = [param.state]

	sim_result = {
		'states' : [],
		'actions' : [],
		'param' : param.to_dict()
		}

	for state in states:

		# state : total number of robots x state dimension per robot 
		# action : total number of robots x action dimension per robot 

		if param.mode == 'GLAS':

			gparam = Gparam() 
			model = DiscreteEmptyNet(gparam, "cpu")
			action = np.zeros((param.num_nodes,9))

			if param.training_team == "a" : 
				model.load_state_dict(torch.load(param.glas_model_A))
				idxs = param.team_1_idxs
			else: 
				model.load_state_dict(torch.load(param.glas_model_B))
				idxs = param.team_2_idxs

			for robot_idx in idxs: 
				o_a,o_b,goal = relative_state(np.array(state),param,robot_idx)
				o_a,o_b,goal = format_data(o_a,o_b,goal)
				action[robot_idx, :] = model(o_a,o_b,goal).detach().numpy().flatten()

			sim_result["states"].append(state) # total number of robots x state dimension per robot 
			sim_result["actions"].append(action) # total number of robots x action dimension per robot 

		else: 

			game_state = state_to_game_state(param,state)
			mctsresult = mctscpp.search(game, game_state, generator, param.tree_size)

			if mctsresult.success: 
				valuePerAction = mctsresult.valuePerAction
				action = value_to_dist(param,valuePerAction) # dim?? 

				sim_result["states"].append(state) 
				sim_result["actions"].append(action) 

		
		sim_result["states"] = np.array(sim_result["states"])
		sim_result["actions"] = np.array(sim_result["actions"])

	dh.write_sim_result(sim_result,param.dataset_fn)

	print('completed instance {}/{}'.format(param.count,param.total))


def get_params(df_param):
	# modified from grun 

	params = []
	seed = int.from_bytes(os.urandom(4), sys.byteorder) # we want the same initial conditions for each case 

	# case = (team_comp, tree_size, training_team, mode)
	# sim_result["action"] in [9 x num_points_per_file]

	curr_ic = -1
	count = 0 
	total = df_param.num_trials * len(df_param.robot_team_composition_cases) * len(df_param.tree_sizes) * len(df_param.training_teams) * len(df_param.modes)
	for trial in range(df_param.num_trials):

		for robot_team_composition in df_param.robot_team_composition_cases:

			df_param.robot_team_composition = robot_team_composition
			df_param.update()
			curr_ic += 1 

			for tree_size in df_param.tree_sizes: 

				for training_team in df_param.training_teams:
					
					for mode in df_param.modes: 

						param = Param()
						param.seed = seed 
						param.num_points_per_file = 1 
						param.rollout_beta = df_param.rollout_beta
						param.glas_model_A = df_param.glas_model_A
						param.glas_model_B = df_param.glas_model_B

						param.robot_team_compositions = df_param.robot_team_composition_cases
						param.modes = df_param.modes 
						param.tree_sizes = df_param.tree_sizes 
						param.training_teams = df_param.training_teams

						param.robot_team_composition = robot_team_composition 
						param.mode = mode 
						param.tree_size = tree_size
						param.trial = trial
						param.training_team = training_team
						param.dataset_fn = "{}/sim_result_{}".format(df_param.data_dir,count)
						param.count = count 
						param.total = total
						param.curr_ic = curr_ic
						count += 1 

						param.update(df_param.state) 
						params.append(param)

	return params

if __name__ == '__main__':

	run_on = True
	parallel_on = False

	df_param = Param()
	df_param.num_trials = 1 # 5
	df_param.rollout_beta = 0.
	df_param.current_results_dir = '../' + df_param.current_results_dir	
	df_param.glas_model_A = '../' + df_param.glas_model_A
	df_param.glas_model_B = '../' + df_param.glas_model_B
	df_param.data_dir = '../../current_results/'
	
	df_param.robot_team_composition_cases = [
		{
		'a': {'standard_robot':1,'evasive_robot':0},
		'b': {'standard_robot':1,'evasive_robot':0}
		},
		# {
		# 'a': {'standard_robot':2,'evasive_robot':0},
		# 'b': {'standard_robot':1,'evasive_robot':0}
		# },
		# {
		# 'a': {'standard_robot':1,'evasive_robot':0},
		# 'b': {'standard_robot':2,'evasive_robot':0}
		# },				
	]
	df_param.training_teams = ["a","b"]
	df_param.modes = ["GLAS", "MCTS_RANDOM", "MCTS_GLAS"]
	df_param.tree_sizes = [1000,5000,10000,50000,100000,500000] 

	if run_on:

		clean_files_containing_str_from_dir('.pickle',df_param.data_dir)

		params = get_params(df_param)

		if parallel_on: 
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(run_sim, params):
				pass 
		else: 
			for param in params: 
				evaluate_stochastic_policy(param)

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.data_dir + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp2_results(sim_results)
	
	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")
