
# standard 
import numpy as np 
import os, glob
import copy
import multiprocessing as mp 
import torch 
import time 

# custom 
from param import Param 
# from learning.discrete_emptynet import DiscreteEmptyNet
from learning.continuous_emptynet import ContinuousEmptyNet
from learning.gaussian_emptynet import GaussianEmptyNet
from learning.value_emptynet import ValueEmptyNet
from learning_interface import format_data, global_to_local, global_to_value, format_data_value
import plotter 
import datahandler as dh

def eval_value(param):

	# exp4_sim_modes:
	# 	- "PREDICT"
	#	 	- if MCTS: plots expected reward at root node corresponding to start state
	# 		- if GLAS: evaluates value model at start state
	# 	- "SIM":
	# 		- plays game, evaluates reward at final state 

	from cpp_interface import play_game, expected_value

	print('running {}/{}'.format(param.count,param.total))

	# save first state 
	nominal_state = copy.deepcopy(param.state)

	# 
	X,Y = discretize_state_space(param.env_xlim,param.env_ylim,param.exp4_dx,param.exp4_dx)

	value_ims = np.nan * np.ones((param.num_nodes,X.shape[0],Y.shape[0]))

	for robot_idx in range(param.num_nodes):

		team_idxs = param.team_1_idxs if param.team == "a" else param.team_2_idxs
		policy_dict = param.policy_dict_a if param.team == "a" else param.policy_dict_b

		if robot_idx not in team_idxs: 
			continue
		
		param.policy_dict = policy_dict
		param.policy_dict["team"] = param.team

		for i_x,x in enumerate(X): 
			for i_y,y in enumerate(Y): 

				param.state[robot_idx] = [x,y,0,0]
				state = np.array(param.state)

				if param.exp4_prediction_type == "PREDICT":

					if policy_dict["sim_mode"] == "MCTS": 
						value = expected_value(param,state,policy_dict,param.team) # query tree 

					if policy_dict["sim_mode"] == "GLAS":

						# format state into value func input 
						n_a = param.num_nodes_A
						n_b = param.num_nodes_B
						n_rg = 0 
						v_a, v_b = global_to_value(param,state)
						v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,n_a,n_b,n_rg)

						# init model 
						model = ValueEmptyNet(param,"cpu")
						model.load_state_dict(torch.load(policy_dict["path_value_fnc"]))

						# call model 
						# value = model(v_a,v_b,n_a,n_b,n_rg) # deterministic 
						_,mu,logvar = model(v_a,v_b,n_a,n_b,n_rg,training=True) # also deterministic 
						value = mu.detach().numpy().squeeze()

				if param.exp4_prediction_type == "SIM":

					temp_sr = play_game(param,param.policy_dict_a,param.policy_dict_b)
					value = temp_sr["rewards"][-1,0]

				value_ims[robot_idx,i_x,i_y] = value 

	sim_result = {
		'X' : X,
		'Y' : Y,
		'value_ims' : value_ims,
		'param' : param.to_dict(),
		'nominal_state' : nominal_state,
	}

	dh.write_sim_result(sim_result,param.dataset_fn)

	print('completed {}/{}'.format(param.count,param.total))


def exp4_get_params(df_param,initial_conditions,robot_team_compositions):

	params = [] 
	count = 0 

	for exp4_prediction_type in df_param.exp4_prediction_types:
		for trial, (initial_condition,robot_team_composition) in enumerate(zip(initial_conditions,robot_team_compositions)):
			for team in ["a","b"]:
				if team == "a": 
					for policy_dict_a in df_param.attackerPolicyDicts:

						param = Param() 
						param.env_l = df_param.env_l
						param.policy_dict_a = policy_dict_a
						param.policy_dict_b = df_param.defenderBaselineDict
						param.exp4_prediction_type = exp4_prediction_type
						param.exp4_sim_modes = df_param.exp4_sim_modes
						param.exp4_prediction_types = df_param.exp4_prediction_types
						param.exp4_dx = df_param.exp4_dx
						param.attackerPolicyDicts = df_param.attackerPolicyDicts
						param.defenderPolicyDicts = df_param.defenderPolicyDicts
						param.trial = trial 
						param.count = count
						param.team = team
						param.robot_team_composition = robot_team_composition
						param.dataset_fn = '{}sim_result_{}'.format(\
							df_param.path_current_results,count)

						param.update(initial_condition=initial_condition)
						params.append(param)
						count += 1 

				elif team == "b":
					for policy_dict_b in df_param.defenderPolicyDicts:

						param = Param() 
						param.env_l = df_param.env_l
						param.policy_dict_a = df_param.attackerBaselineDict
						param.policy_dict_b = policy_dict_b
						param.exp4_prediction_type = exp4_prediction_type
						param.exp4_sim_modes = df_param.exp4_sim_modes
						param.exp4_prediction_types = df_param.exp4_prediction_types
						param.exp4_dx = df_param.exp4_dx
						param.attackerPolicyDicts = df_param.attackerPolicyDicts
						param.defenderPolicyDicts = df_param.defenderPolicyDicts
						param.trial = trial 
						param.count = count
						param.team = team
						param.robot_team_composition = robot_team_composition
						param.dataset_fn = '{}sim_result_{}'.format(\
							df_param.path_current_results,count)

						param.update(initial_condition=initial_condition)
						params.append(param)
						count += 1 

	total = count 
	for param in params: 
		param.total = total

	return params


def discretize_state_space(env_xlim,env_ylim,dx,dy):
	X = np.arange(env_xlim[0],env_xlim[1],dx) # + dx / 2.0
	Y = np.arange(env_ylim[0],env_ylim[1],dy) # + dy / 2.0 
	return X,Y

def exp4_make_games(df_param):	
	initial_conditions,robot_team_compositions = [],[]
	for robot_team_composition in df_param.robot_team_compositions:
		df_param.robot_team_composition = robot_team_composition
		df_param.update()
		for trial in range(df_param.exp4_num_trials):
			initial_conditions.append(df_param.make_initial_condition())
			robot_team_compositions.append(robot_team_composition)

	return initial_conditions,robot_team_compositions

def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)


def main():

	sim_parallel_on = True 
	run_on = True	
	model_dir = '../current/models'

	df_param = Param()
	df_param.exp4_prediction_types = ["PREDICT","SIM"] 
	df_param.exp4_sim_modes = ["MCTS","GLAS"] 
	df_param.exp4_max_policy = 1
	df_param.exp4_dx = 0.05
	df_param.exp4_num_trials = 2
	df_param.exp4_tree_sizes = [100] 
	df_param.l_num_expert_nodes = 100

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
	for policy_i in range(df_param.exp4_max_policy+1):
		for policy_dicts in [df_param.attackerPolicyDicts,df_param.defenderPolicyDicts]:
			for tree_size in df_param.exp4_tree_sizes: 
				policy_dicts.append({
					'sim_mode' : 				"MCTS",
					'path_glas_model_a' : 		'{}/a{}.pt'.format(model_dir,policy_i) if policy_i > 0  else None,
					'path_glas_model_b' : 		'{}/b{}.pt'.format(model_dir,policy_i) if policy_i > 0  else None, 
					'path_value_fnc' : 			'{}/v{}.pt'.format(model_dir,policy_i) if policy_i > 0  else None, 
					'mcts_tree_size' : 			tree_size,
					'mcts_rollout_horizon' : 	df_param.rollout_horizon,
					'mcts_c_param' : 			df_param.l_mcts_c_param,
					'mcts_pw_C' : 				df_param.l_mcts_pw_C,
					'mcts_pw_alpha' : 			df_param.l_mcts_pw_alpha,
					'mcts_beta1' : 				df_param.l_mcts_beta1,
					'mcts_beta2' : 				df_param.l_mcts_beta2,
					'mcts_beta3' : 				df_param.l_mcts_beta3,
					})

			if policy_i > 0:
				policy_dicts.append({
					'sim_mode' : 				"GLAS",
					'path_glas_model' : 		'{}/a{}.pt'.format(model_dir,policy_i),
					'path_value_fnc' : 			'{}/v{}.pt'.format(model_dir,policy_i),
					'deterministic': 			True,
				})

	# games 
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

	initial_conditions, robot_team_compositions = exp4_make_games(df_param) 

	if run_on: 
		format_dir(df_param)
		params = exp4_get_params(df_param,initial_conditions,robot_team_compositions)

		if sim_parallel_on: 	
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(eval_value, params):
				pass 
		else:
			for param in params: 
				eval_value(param)	

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	print('plotting...')
	plotter.plot_exp4_results(sim_results)
	
	print('saving and opening figs...')
	plotter.save_figs("plots/exp4.pdf")
	plotter.open_figs("plots/exp4.pdf")

if __name__ == '__main__':
	main()