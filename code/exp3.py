
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
	
	print('playing game {}/{}'.format(param.count+1,param.total))
	sim_result = play_game(param,param.policy_dict_a,param.policy_dict_b)
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
		for policy_dict_a in df_param.attackerPolicyDicts:
			for policy_dict_b in df_param.defenderPolicyDicts:

				param = Param() 
				param.env_l = df_param.env_l
				param.policy_dict_a = policy_dict_a
				param.policy_dict_b = policy_dict_b
				param.attackerPolicyDicts = df_param.attackerPolicyDicts
				param.defenderPolicyDicts = df_param.defenderPolicyDicts
				param.trial = trial 
				param.count = count
				param.total = total
				param.robot_team_composition = robot_team_composition
				param.dataset_fn = '{}sim_result_{:03d}'.format(\
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

	df_param.env_l = 5.0
	df_param.init_on_sides = True
	df_param.num_trials = 100
	max_policy_file = 6
	skip_policy_file = 2
	glas_policy_files = range(1,max_policy_file+1,skip_policy_file)
	mcts_policy_files = range(1,max_policy_file+1,skip_policy_file)
	name = "current/models" # "saved/t11/models" 
	# name = "saved/3d_dubins_1/models" 

	mode = 1

	if mode == 0:

		df_param.attackerPolicyDicts = []
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : "PANAGOU"
			}])
		df_param.attackerPolicyDicts.extend([{
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
			}])
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,mcts_policy_files[-1]),
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,mcts_policy_files[-1]),
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,mcts_policy_files[-1]),
			'mcts_tree_size' : 			df_param.l_num_expert_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])	
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,i) if i > 0  else None,
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,i) if i > 0  else None,
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,i) if i > 0  else None,
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
		} for i in mcts_policy_files])
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"GLAS",
			'path_glas_model' : 		'../{}/a{}.pt'.format(name,i),
			'deterministic': 			True,
		} for i in glas_policy_files])

		df_param.defenderPolicyDicts = []
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : "PANAGOU"
			}])
		df_param.defenderPolicyDicts.extend([{
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
			}])	
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,mcts_policy_files[-1]),
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,mcts_policy_files[-1]),
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,mcts_policy_files[-1]),
			'mcts_tree_size' : 			df_param.l_num_expert_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])		
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,i) if i > 0  else None,
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,i) if i > 0  else None,
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,i) if i > 0  else None,
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			} for i in mcts_policy_files])
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"GLAS",
			'path_glas_model' : 		'../{}/b{}.pt'.format(name,i),
			'deterministic': 			True,
			} for i in glas_policy_files])

	elif mode == 1 :

		df_param.attackerPolicyDicts = []
		df_param.attackerPolicyDicts.extend([{
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
			}])
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		None, 
			'path_glas_model_b' : 		None, 
			'path_value_fnc' : 			None, 
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])			
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,mcts_policy_files[-1]),
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,mcts_policy_files[-1]),
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,mcts_policy_files[-1]),
			'mcts_tree_size' : 			df_param.l_num_expert_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])	
		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,i) if i > 0  else None,
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,i) if i > 0  else None,
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,i) if i > 0  else None,
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
		} for i in mcts_policy_files])

		df_param.defenderPolicyDicts = []
		df_param.defenderPolicyDicts.extend([{
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
			}])
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		None, 
			'path_glas_model_b' : 		None, 
			'path_value_fnc' : 			None, 
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])			
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,mcts_policy_files[-1]),
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,mcts_policy_files[-1]),
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,mcts_policy_files[-1]),
			'mcts_tree_size' : 			df_param.l_num_expert_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			}])		
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"D_MCTS",
			'path_glas_model_a' : 		'../{}/a{}.pt'.format(name,i) if i > 0  else None,
			'path_glas_model_b' : 		'../{}/b{}.pt'.format(name,i) if i > 0  else None,
			'path_value_fnc' : 			'../{}/v{}.pt'.format(name,i) if i > 0  else None,
			'mcts_tree_size' : 			df_param.l_num_learner_nodes,
			'mcts_rollout_horizon' : 	df_param.rollout_horizon,
			'mcts_c_param' : 			df_param.l_mcts_c_param,
			'mcts_pw_C' : 				df_param.l_mcts_pw_C,
			'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
			'mcts_beta1' : 				df_param.l_mcts_beta1,
			'mcts_beta2' : 				df_param.l_mcts_beta2,
			'mcts_beta3' : 				df_param.l_mcts_beta3,
			} for i in mcts_policy_files])

	elif mode == 2: 

		df_param.attackerPolicyDicts = []
		df_param.defenderPolicyDicts = []

		df_param.attackerPolicyDicts.extend([{
			'sim_mode' : 				"GLAS",
			'path_glas_model' : 		'../{}/a{}.pt'.format(name,i),
			'deterministic': 			True,
		} for i in glas_policy_files])
		df_param.defenderPolicyDicts.extend([{
			'sim_mode' : 				"GLAS",
			'path_glas_model' : 		'../{}/b{}.pt'.format(name,i),
			'deterministic': 			True,
			} for i in glas_policy_files])


	parser = argparse.ArgumentParser()
	parser.add_argument("-game_file", default=None, required=False)
	args = parser.parse_args()

	if not args.game_file is None: 
		initial_conditions,robot_team_compositions = read_games_file(args.game_file)
	else: 
		df_param.robot_team_compositions = [
			{
			'a': {'standard_robot':2,'evasive_robot':0},
			'b': {'standard_robot':2,'evasive_robot':0}
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
	files = sorted(glob.glob(df_param.path_current_results + '/*'))
	for sim_result_dir in files :
		sim_results.append(dh.load_sim_result(sim_result_dir))

	# Plot results of all the runs
	print("\nGenerating Plots")
	plotter.plot_exp3_results(sim_results)

	# Plot results of each run
	count = 0 
	for sim_result in sim_results:
		_, filename = os.path.split(files[count])

		# Plot results of each run
		plotter.plot_tree_results(sim_result,\
			title='Game: {} - File: {} - Time: {:5.2f} s\nA (Blue):{}\n B (Orange):{}'.format(\
				sim_result["param"]["trial"],
				filename,
				max(sim_result["times"]),
				policy_to_label(sim_result["param"]["policy_dict_a"]),\
				policy_to_label(sim_result["param"]["policy_dict_b"])))
		count += 1 
		# Limit the maximum number of results files to plot
		if count >= 5: 
			break 

	print('saving and opening figs...')
	plotter.save_figs("plots/exp3.pdf")
	plotter.open_figs("plots/exp3.pdf")

