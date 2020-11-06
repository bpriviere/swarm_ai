


import time 
import os, sys, glob
import argparse
import multiprocessing as mp
import yaml 
from collections import defaultdict

import datahandler as dh 
import plotter 
from plotter import policy_to_label
from param import Param

def wrap_play_game(param):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import play_game
	
	print('playing game {}/{}'.format(param.count,param.total))
	sim_result = play_game(param,param.policy_dict_a,param.policy_dict_b)
	dh.write_sim_result(sim_result,param.dataset_fn)


def make_games(df_param):

	initial_conditions,robot_team_compositions = [],[]
	for robot_team_composition in df_param.robot_team_compositions:
		df_param.robot_team_composition = robot_team_composition
		df_param.update()
		for trial in range(df_param.num_trials):
			initial_conditions.append(df_param.make_initial_condition())
			robot_team_compositions.append(robot_team_composition)

	return initial_conditions,robot_team_compositions


def get_params(df_param,initial_conditions,robot_team_compositions):

	params = [] 
	count = 0 

	for trial, (initial_condition,robot_team_composition) in enumerate(zip(initial_conditions,robot_team_compositions)):
		for test_team in ["a","b"]:
			if test_team == "a":
				for policy_dict_a in df_param.attackerPolicyDicts:
					param = Param() 
					param.env_l = df_param.env_l
					param.policy_dict_a = policy_dict_a
					param.policy_dict_b = df_param.defenderBaselineDict
					param.tree_sizes = df_param.tree_sizes
					param.test_team = test_team
					param.trial = trial 
					param.count = count
					param.robot_team_composition = robot_team_composition
					param.dataset_fn = '{}sim_result_{}'.format(\
						df_param.path_current_results,count)
					param.update(initial_condition=initial_condition)
					params.append(param)
					count += 1 
			if test_team == "b":
				for policy_dict_b in df_param.defenderPolicyDicts:
					param = Param() 
					param.env_l = df_param.env_l
					param.policy_dict_a = df_param.attackerBaselineDict
					param.policy_dict_b = policy_dict_b
					param.tree_sizes = df_param.tree_sizes
					param.test_team = test_team
					param.trial = trial 
					param.count = count
					param.robot_team_composition = robot_team_composition
					param.dataset_fn = '{}sim_result_{}'.format(\
						df_param.path_current_results,count)
					param.update(initial_condition=initial_condition)
					params.append(param)
					count += 1 
	
	for param in params:
		param.total = count 

	return params

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

if __name__ == '__main__':

	# params 
	df_param = Param()
	df_param.max_policy = 1
	df_param.policies = [1,3,5] # list(range(df_param.max_policy+1))
	df_param.tree_sizes = [100,,500,1000,10000,50000] # ,1000,10000,50000] #1000] #,5000,10000]
	df_param.num_trials = 100
	df_param.c_params = [1.0,1.4,2.0,5.0] 
	name = "../current/models/" #"../saved/r28"

	# attackers 
	df_param.attackerBaselineDict = {
		'sim_mode' : 				"MCTS",
		'path_glas_model_a' : 		None,
		'path_glas_model_b' : 		None, 
		'path_value_fnc' : 			None, 
		'mcts_tree_size' : 			df_param.l_num_expert_nodes,
		'mcts_rollout_horizon' : 	df_param.rollout_horizon,
		'mcts_c_param' : 			df_param.l_mcts_c_param,
		'mcts_pw_C' : 				df_param.l_mcts_pw_C,
		'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
		'mcts_beta1' : 				df_param.l_mcts_beta1,
		'mcts_beta2' : 				df_param.l_mcts_beta2,
		'mcts_beta3' : 				df_param.l_mcts_beta3,
	}
	df_param.defenderBaselineDict = df_param.attackerBaselineDict.copy()

	df_param.attackerPolicyDicts = []
	df_param.defenderPolicyDicts = []
	for policy_i in df_param.policies:
		for tree_size in df_param.tree_sizes: 
			for c_param in df_param.c_params: 
				for policy_dicts in [df_param.attackerPolicyDicts,df_param.defenderPolicyDicts]:
					policy_dicts.append({
						'sim_mode' : 				"MCTS",
						'path_glas_model_a' : 		'{}/a{}.pt'.format(name,policy_i) if policy_i > 0  else None,
						'path_glas_model_b' : 		'{}/b{}.pt'.format(name,policy_i) if policy_i > 0  else None, 
						'path_value_fnc' : 			'{}/v{}.pt'.format(name,policy_i) if policy_i > 0  else None, 
						'mcts_tree_size' : 			tree_size,
						'mcts_rollout_horizon' : 	df_param.rollout_horizon,
						'mcts_c_param' : 			c_param,
						'mcts_pw_C' : 				df_param.l_mcts_pw_C,
						'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
						'mcts_beta1' : 				df_param.l_mcts_beta1,
						'mcts_beta2' : 				df_param.l_mcts_beta2,
						'mcts_beta3' : 				df_param.l_mcts_beta3,
						})

	# games 
	df_param.robot_team_compositions = [
		{
		'a': {'standard_robot':2,'evasive_robot':0},
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
	initial_conditions,robot_team_compositions = make_games(df_param)	

	run_on = True
	if run_on: 

		params = get_params(df_param,initial_conditions,robot_team_compositions)
		format_dir(df_param)

		if df_param.sim_parallel_on: 	
			with mp.Pool(mp.cpu_count()-1) as pool:
				for _ in pool.imap_unordered(wrap_play_game, params):
					pass
		else:
			for param in params: 
				wrap_play_game(param) 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp7_results(sim_results)

	print('saving and opening figs...')
	plotter.save_figs("plots/exp7.pdf")
	plotter.open_figs("plots/exp7.pdf")

