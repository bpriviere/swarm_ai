
# standard
import numpy as np 
import multiprocessing as mp
import glob
import os 
import shutil 
import subprocess
import sys
import torch

# ours
from param import Param 
import datahandler as dh
import plotter 
from cpp_interface import evaluate_expert, evaluate_glas
from mice import relative_state, format_data
from learning.discrete_emptynet import DiscreteEmptyNet


def evaluate_stochastic_policy(param):

	print('running param {}/{}'.format(param.count,param.total))

	if 'MCTS' in param.sim_mode:
		evaluate_expert([param.state],param)

	elif param.sim_mode == 'GLAS':
		
		state = param.state
		model = DiscreteEmptyNet(param, "cpu")
		action = np.zeros((param.num_nodes,9))

		sim_result = {
			'states' : [],
			'actions' : [],
			'param' : param.to_dict()
			}

		if param.training_team == "a" : 
			model.load_state_dict(torch.load(param.path_glas_model_a))
			idxs = param.team_1_idxs
		else: 
			model.load_state_dict(torch.load(param.path_glas_model_b))
			idxs = param.team_2_idxs

		for robot_idx in idxs: 
			o_a,o_b,goal = relative_state(np.array(state),param,robot_idx)
			o_a,o_b,goal = format_data(o_a,o_b,goal)
			action[robot_idx, :] = model(o_a,o_b,goal).detach().numpy().flatten()

		sim_result["states"].append(state) # total number of robots x state dimension per robot 
		sim_result["actions"].append(action) # total number of robots x action dimension per robot 

		sim_result["states"] = np.array(sim_result["states"])
		sim_result["actions"] = np.array(sim_result["actions"])

		dh.write_sim_result(sim_result,param.dataset_fn)

	else:
		exit('mode not recognized',param.sim_mode)


def get_params(df_param):

	params = []
	seed = int.from_bytes(os.urandom(4), sys.byteorder) 
	curr_ic = -1
	count = 0 
	total = df_param.sim_num_trials*len(df_param.l_robot_team_composition_cases)*len(df_param.mcts_tree_sizes)*\
		len(df_param.l_training_teams)*len(df_param.sim_modes)

	for trial in range(df_param.sim_num_trials):
		for robot_team_composition in df_param.l_robot_team_composition_cases:

			df_param.robot_team_composition = robot_team_composition
			df_param.update() # makes an initial condition 
			curr_ic += 1 

			for tree_size in df_param.mcts_tree_sizes: 
				for training_team in df_param.l_training_teams:
					for mode in df_param.sim_modes: 

						param = Param()
						param.seed = seed 

						param.mcts_rollout_beta = df_param.mcts_rollout_beta
						param.mcts_tree_sizes = df_param.mcts_tree_sizes 
						param.l_robot_team_compositions = df_param.l_robot_team_composition_cases
						param.sim_modes = df_param.sim_modes 
						param.l_training_teams = df_param.l_training_teams
						param.l_num_points_per_file = 1 
						param.sim_num_trials = df_param.sim_num_trials

						param.robot_team_composition = robot_team_composition 
						param.sim_trial = trial
						param.mcts_tree_size = tree_size
						param.sim_mode = mode 
						param.training_team = training_team
						param.dataset_fn = '{}sim_result_{}'.format(df_param.path_current_results,count)
						param.count = count 
						param.total = total
						param.curr_ic = curr_ic
						count += 1 

						param.update(df_param.state) 
						params.append(param)

	return params

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

if __name__ == '__main__':

	df_param = Param()
	df_param.sim_modes = ["GLAS","MCTS_RANDOM","MCTS_GLAS"]
	df_param.mcts_tree_sizes = [1000,5000,10000,50000,100000] 

	run_on = True
	if run_on: 

		format_dir(df_param)
		params = get_params(df_param)

		if df_param.sim_parallel_on: 
			pool = mp.Pool(min(len(params), mp.cpu_count()-1))
			for _ in pool.imap_unordered(evaluate_stochastic_policy, params):
				pass 
		else: 
			for param in params: 
				evaluate_stochastic_policy(param)

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp2_results(sim_results)
	
	plotter.save_figs("plots.pdf")
	plotter.open_figs("plots.pdf")
