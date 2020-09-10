
# standard
import numpy as np 
import multiprocessing as mp
import glob
import os 
import shutil 
import subprocess
import sys
import torch
import argparse

# ours
from param import Param 
import datahandler as dh
import plotter 
from cpp_interface import evaluate_expert
from mice import relative_state, format_data
from learning.discrete_emptynet import DiscreteEmptyNet


def evaluate_stochastic_policy(param):

	print('running param {}/{}'.format(param.count,param.total))

	if 'MCTS' in param.sim_mode:
		evaluate_expert([param.state],param)

	elif param.sim_mode == 'GLAS':
		
		sim_result = {
			'states' : [],
			'actions' : [],
			'param' : param.to_dict()
			}
		state = param.state
		action = np.zeros((param.num_nodes,9))

		model = DiscreteEmptyNet(param, "cpu")
		if param.training_team == "a" : 
			model.load_state_dict(torch.load(param.path_glas_model_a))
			idxs = param.team_1_idxs
		else: 
			model.load_state_dict(torch.load(param.path_glas_model_b))
			idxs = param.team_2_idxs

		for robot_idx in idxs: 
			o_a,o_b,goal = relative_state(np.array(state),param,robot_idx)
			o_a,o_b,goal = format_data(o_a,o_b,goal)
			value, policy = model(o_a,o_b,goal)
			action[robot_idx, :] = policy.detach().numpy().flatten()

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
	total = df_param.num_trials*len(df_param.robot_team_compositions)*len(df_param.mcts_tree_sizes)*\
		len(df_param.training_teams)*len(df_param.modes)
	num_ics = df_param.num_trials*len(df_param.robot_team_compositions)

	for robot_team_composition in df_param.robot_team_compositions:
		
		df_param.robot_team_composition = robot_team_composition
		df_param.update()

		for trial in range(df_param.num_trials):

			initial_condition = df_param.make_initial_condition()
			curr_ic += 1 

			for tree_size in df_param.mcts_tree_sizes: 
				for training_team in df_param.training_teams:
					for mode in df_param.modes: 

						param = Param(seed=seed)

						# global param 
						param.path_glas_model_a = df_param.path_glas_model_a
						param.path_glas_model_b = df_param.path_glas_model_b
						param.mcts_tree_sizes = df_param.mcts_tree_sizes 
						param.robot_team_compositions = df_param.robot_team_compositions
						param.modes = df_param.modes
						param.training_teams = df_param.training_teams
						param.l_num_points_per_file = 1 
						param.sim_num_trials = df_param.num_trials
						param.num_ics = num_ics

						# local param 
						if mode == "GLAS":
							param.mcts_rollout_beta = 0.0
							param.sim_mode = "GLAS"
						elif "MCTS" in mode:
							param.mcts_rollout_beta = float(mode.split("MCTS ")[-1])
							param.sim_mode = "MCTS"

						param.mode = mode  
						param.robot_team_composition = robot_team_composition 
						param.sim_trial = trial
						param.mcts_tree_size = tree_size
						param.training_team = training_team
						param.dataset_fn = '{}sim_result_{}'.format(df_param.path_current_results,count)
						param.count = count 
						param.total = total
						param.curr_ic = curr_ic
						count += 1 

						param.update(initial_condition=initial_condition) 
						params.append(param)

	return params

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

if __name__ == '__main__':

	df_param = Param()
	df_param.num_trials = 1
	df_param.modes = ["GLAS", "MCTS 0.0", "MCTS 0.5", "MCTS 1.0"]
	df_param.mcts_tree_sizes = [1000,5000,10000] #,50000,100000,500000] 
	# df_param.path_glas_model_a = "../saved/IL_4/models/a8.pt"
	# df_param.path_glas_model_b = "../saved/IL_4/models/b8.pt"
	df_param.robot_team_compositions = [
		{
		'a': {'standard_robot':2,'evasive_robot':0},
		'b': {'standard_robot':1,'evasive_robot':0}
		},
		]
	df_param.training_teams = ["a"] #["a","b"]	

	parser = argparse.ArgumentParser()
	parser.add_argument("-path_glas_model_a", default=None, required=False)
	parser.add_argument("-path_glas_model_b", default=None, required=False)
	args = parser.parse_args() 	

	if not args.path_glas_model_a is None: 
		df_param.path_glas_model_a = args.path_glas_model_a
	if not args.path_glas_model_b is None: 
		df_param.path_glas_model_b = args.path_glas_model_b	

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
	
	plotter.save_figs("plots/exp2.pdf")
	plotter.open_figs("plots/exp2.pdf")
