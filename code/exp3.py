
import time 
import os, sys, glob
import argparse
import multiprocessing as mp
from collections import defaultdict

import datahandler as dh 
import plotter 
from plotter import policy_to_label
from param import Param
from cpp_interface import rollout , play_game

def get_params(df_param):

	params = [] 
	count = 0 
	total = len(df_param.robot_team_compositions)*len(df_param.attackerPolicyDicts)*\
		len(df_param.defenderPolicyDicts)*df_param.num_trials
	seed = int.from_bytes(os.urandom(4), sys.byteorder)

	for robot_team_composition in df_param.robot_team_compositions:
		for trial in range(df_param.num_trials):

			df_param.robot_team_composition = robot_team_composition
			df_param.update()
			state = df_param.make_initial_condition()

			for policy_a_dict in df_param.attackerPolicyDicts:
				for policy_b_dict in df_param.defenderPolicyDicts:

					param = Param(seed=seed) 
					param.policy_a_dict = policy_a_dict
					param.policy_b_dict = policy_b_dict
					param.attackerPolicyDicts = df_param.attackerPolicyDicts
					param.defenderPolicyDicts = df_param.defenderPolicyDicts
					param.trial = trial
					param.count = count
					param.total = total
					param.robot_team_composition = robot_team_composition
					param.dataset_fn = '{}sim_result_{}'.format(\
						df_param.path_current_results,count)
					param.update(initial_condition=state)
					params.append(param)
					count += 1 

	return params

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)

if __name__ == '__main__':

	df_param = Param()

	df_param.num_trials = 20

	df_param.robot_team_compositions = [
		{
		'a': {'standard_robot':1,'evasive_robot':0},
		'b': {'standard_robot':1,'evasive_robot':0}
		},
		{
		'a': {'standard_robot':2,'evasive_robot':0},
		'b': {'standard_robot':1,'evasive_robot':0}
		},
		# {
		# 'a': {'standard_robot':1,'evasive_robot':0},
		# 'b': {'standard_robot':2,'evasive_robot':0}
		# }
	]
	
	df_param.attackerPolicyDicts = [
		{
			"sim_mode" : "GLAS",
			"path_glas_model_a" : "../saved/IL/models/a0.pt",
			"mcts_tree_size" : 1000,
		},
		{
			"sim_mode" : "GLAS",
			"path_glas_model_a" : "../saved/IL/models/a1.pt",
			"mcts_tree_size" : 10000,
		},
		{
			"sim_mode" : "GLAS",
			"path_glas_model_a" : "../saved/IL/models/a2.pt",
			"mcts_tree_size" : 50000,
		},
		{
			"sim_mode" : "GLAS",
			"path_glas_model_a" : "../saved/IL/models/a3.pt",
			"mcts_tree_size" : 100000,
		},
	]

	df_param.defenderPolicyDicts = [
		{
			"sim_mode" : "GLAS",
			"path_glas_model_b" : "../saved/IL/models/b0.pt",
			"mcts_rollout_beta" : 0.0, 
			"mcts_tree_size" : 100000,
		},				
		{
			"sim_mode" : "GLAS",
			"path_glas_model_b" : "../saved/IL/models/b1.pt",
			"mcts_rollout_beta" : 0.0, 
			"mcts_tree_size" : 1000,
		},
		{
			"sim_mode" : "GLAS",
			"path_glas_model_b" : "../saved/IL/models/b2.pt",
			"mcts_rollout_beta" : 0.0, 
			"mcts_tree_size" : 10000,
		},
		{
			"sim_mode" : "GLAS",
			"path_glas_model_b" : "../saved/IL/models/b3.pt",
			"mcts_rollout_beta" : 0.0, 
			"mcts_tree_size" : 50000,
		},
	]

	run_on = True
	if run_on: 

		params = get_params(df_param)
		format_dir(df_param)

		if df_param.sim_parallel_on: 	
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(play_game, params):
				pass 
		else:
			for param in params: 
				play_game(param) 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp3_results(sim_results)

	# for sim_result in sim_results:
	# 	plotter.plot_tree_results(sim_result,title='T: {}, A:{}, B:{}'.format(\
	# 		sim_result["param"]["trial"],
	# 		policy_to_label(sim_result["param"]["policy_a_dict"]),\
	# 		policy_to_label(sim_result["param"]["policy_b_dict"])))

	print('saving and opening figs...')
	plotter.save_figs("plots/exp3.pdf")
	plotter.open_figs("plots/exp3.pdf")

