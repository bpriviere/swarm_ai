


import time 
import os, sys, glob
import argparse
import multiprocessing as mp
import yaml 
from collections import defaultdict
import numpy as np 

import datahandler as dh 
import plotter 
from plotter import policy_to_label
from param import Param


def wrap_expected_value(param):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import expected_value, self_play

	print('expected_value {}/{}'.format(param.count,param.total))
	value, best_action = expected_value(param,param.state,param.policy_dict,param.team)

	sim_result = {
		"param" : param.to_dict() , 
		"value" : value , 
		# "best_action" : best_action,
	}
	dh.write_sim_result(sim_result,param.dataset_fn)


def get_params_exp8(df_param):

	# print('df_param.tree_sizes',df_param.tree_sizes)

	params = [] 
	count = 0 
	for policy_dict in df_param.policy_dicts: 
		for tree_size in df_param.tree_sizes: 
			for i_trial in range(df_param.num_trials): 
				param = Param() 
				param.count = count 
				param.policy_dict = policy_dict.copy()
				# print('tree_size',tree_size)
				param.policy_dict["mcts_tree_size"] = tree_size 
				# print('param.policy_dict["mcts_tree_size"]',param.policy_dict["mcts_tree_size"])
				param.dataset_fn = '{}sim_result_{:03d}'.format(\
					df_param.path_current_results,count)
				param.update(initial_condition=df_param.state)
				# print('param.policy_dict["mcts_tree_size"]',param.policy_dict["mcts_tree_size"])				
				params.append(param)
				count += 1 

	for param in params: 
		param.total = count 
		param.team = df_param.team 
		# print('param.policy_dict["mcts_tree_size"]',param.policy_dict["mcts_tree_size"])

	return params

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

if __name__ == '__main__':

	# goal: calculate root node value for fixed initial condition
	# compare with tree size and with and without heuristics 

	# params 
	df_param = Param()
	df_param.tree_sizes = [100,1000,5000,10000,50000,100000,500000,1000000] #,1000,10000,50000] # ,1000,10000,50000] #1000] #,5000,10000]
	df_param.num_trials = 20

	# policy dicts 
	path_to_models = '../current/models'
	model_number = 11
	df_param.policy_dicts = [df_param.policy_dict.copy() for i in range(2)] #,df_param.policy_dict.copy()]
	df_param.policy_dicts[1]["path_glas_model_a"] = "{}/a{}.pt".format(path_to_models,model_number)
	df_param.policy_dicts[1]["path_glas_model_b"] = "{}/b{}.pt".format(path_to_models,model_number)
	df_param.policy_dicts[1]["path_glas_model_v"] = "{}/v{}.pt".format(path_to_models,model_number)

	# games 
	df_param.robot_team_compositions = [
		{
		'a': {'standard_robot':1,'evasive_robot':0},
		'b': {'standard_robot':1,'evasive_robot':0}
		},
	]
	df_param.state = df_param.env_l*np.array( [ \
		[   0.1,   0.50,   0.000,   0.000 ], \
		# [   0.20,   0.35,   0.000,   0.000 ], \
		[   0.8,   0.50,   0.000,   0.000 ] ])
	df_param.team = "a" 

	run_on = True
	if run_on: 

		params = get_params_exp8(df_param)
		format_dir(df_param)

		if df_param.sim_parallel_on: 	
			with mp.Pool(mp.cpu_count()-1) as pool:
				for _ in pool.imap_unordered(wrap_expected_value, params):
					pass
		else:
			for param in params: 
				wrap_expected_value(param) 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp8_results(sim_results)

	print('play one game...')
	from cpp_interface import self_play
	sim_result_game = self_play(params[0])
	plotter.plot_tree_results(sim_result_game)


	print('saving and opening figs...')
	plotter.save_figs("plots/exp8.pdf")
	plotter.open_figs("plots/exp8.pdf")