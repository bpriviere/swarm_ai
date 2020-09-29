
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
	sim_result = play_game(param,param.policy_a_dict,param.policy_b_dict)
	dh.write_sim_result(sim_result,param.dataset_fn)


def read_games_file(games_file):

	with open(games_file) as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)

	initial_conditions,robot_team_compositions = [],[]
	for game in cfg["games"]:

		initial_condition = []
		robot_team_composition = dict()

		robot_team_composition["a"] = dict()
		robot_team_composition["b"] = dict()

		for robot in game["team_a"]:
			if robot["type"] not in robot_team_composition["a"].keys():
				robot_team_composition["a"][robot["type"]] = 0 
			initial_condition.append(robot["x0"])
			robot_team_composition["a"][robot["type"]] += 1

		for robot in game["team_b"]:
			if robot["type"] not in robot_team_composition["b"].keys():
				robot_team_composition["b"][robot["type"]] = 0 
			initial_condition.append(robot["x0"])
			robot_team_composition["b"][robot["type"]] += 1	

		initial_conditions.append(initial_condition)
		robot_team_compositions.append(robot_team_composition)

	return initial_conditions,robot_team_compositions


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
	total = len(initial_conditions)*len(df_param.attackerPolicyDicts)*\
		len(df_param.defenderPolicyDicts)

	for trial, (initial_condition,robot_team_composition) in enumerate(zip(initial_conditions,robot_team_compositions)):
		for policy_a_dict in df_param.attackerPolicyDicts:
			for policy_b_dict in df_param.defenderPolicyDicts:

				param = Param() 
				param.env_l = df_param.env_l
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

				param.update(initial_condition=initial_condition)
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

	df_param.env_l = 1.0

	df_param.attackerPolicyDicts = [
		{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../current/models/a0.pt',
			'path_glas_model_b' : 		'../current/models/b0.pt', 
			'mcts_tree_size' : 			50000,
			'mcts_rollout_horizon' : 	100,	
			'mcts_rollout_beta' : 		0.0,
			'mcts_c_param' : 			0.5,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_vf_beta' : 			0.0,
		},				
		{
			"sim_mode" : "PANAGOU",
		},		
	]

	df_param.defenderPolicyDicts = [				
		{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../current/models/a0.pt',
			'path_glas_model_b' : 		'../current/models/b0.pt', 
			'mcts_tree_size' : 			50000,
			'mcts_rollout_horizon' : 	100,	
			'mcts_rollout_beta' : 		0.0,
			'mcts_c_param' : 			0.5,
			'mcts_pw_C' : 				1.0,
			'mcts_pw_alpha' : 			0.25,
			'mcts_vf_beta' : 			0.0,
		},	
		{
			"sim_mode" : "PANAGOU",
		},
	]

	parser = argparse.ArgumentParser()
	parser.add_argument("-game_file", default=None, required=False)
	args = parser.parse_args()

	if not args.game_file is None: 
		initial_conditions,robot_team_compositions = read_games_file(args.game_file)
	else: 
		df_param.num_trials = 2
		df_param.robot_team_compositions = [
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
		initial_conditions,robot_team_compositions = make_games(df_param)

	run_on = True
	if run_on: 

		params = get_params(df_param,initial_conditions,robot_team_compositions)
		format_dir(df_param)

		if df_param.sim_parallel_on: 	
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(wrap_play_game, params):
				pass 
		else:
			for param in params: 
				wrap_play_game(param) 

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp3_results(sim_results)

	count = 0 
	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result,title='T: {}, A:{}, B:{}'.format(\
			sim_result["param"]["trial"],
			policy_to_label(sim_result["param"]["policy_a_dict"]),\
			policy_to_label(sim_result["param"]["policy_b_dict"])))
		count += 1 
		if count >= 10:
			break 

	print('saving and opening figs...')
	plotter.save_figs("plots/exp3.pdf")
	plotter.open_figs("plots/exp3.pdf")

