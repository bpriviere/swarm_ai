

# plot trees across the state space 
# plots heuristics across the state space 


# custom 
from param import Param 
from learning.value_emptynet import ValueEmptyNet
from learning.policy_emptynet import PolicyEmptyNet
from learning_interface import format_data_value, global_to_value , global_to_local, format_data
from param import Param 
import plotter 

# standard 
import torch 
import numpy as np 
import copy 


def evaluate_values(param,states):

	values = np.zeros((states.shape[0],states.shape[1]))

	cn_a = len(param.team_1_idxs)
	cn_b = len(param.team_2_idxs)
	cn_rg = 0 

	with torch.no_grad():
		model = ValueEmptyNet(param,"cpu")
		model.load_state_dict(torch.load(param.policy_dict["path_value_fnc"]))

		for i_x in range(states.shape[0]):
			for i_y in range(states.shape[1]):
				state = states[i_x,i_y,:]

				v_a,v_b = global_to_value(param,state)
				v_a,v_b,n_a,n_b,n_rg = format_data_value(v_a,v_b,cn_a,cn_b,cn_rg)
				_,mu,logvar = model(v_a,v_b,n_a,n_b,n_rg,training=True)

				values[i_x,i_y] = mu.detach().numpy().squeeze()

	return values 	
	
def evaluate_policy(param,states):

	policies = np.zeros((states.shape[0],states.shape[1],2))

	with torch.no_grad():
		model = PolicyEmptyNet(param,"cpu")
		model.load_state_dict(torch.load(param.policy_dict["path_glas_model_a"]))

		for i_x in range(states.shape[0]):
			for i_y in range(states.shape[1]):
				state = states[i_x,i_y,:]
				o_a,o_b,relative_goal = global_to_local(state,param,0)
				o_a,o_b,relative_goal = format_data(o_a,o_b,relative_goal)
				_,mu,logvar = model(o_a,o_b,relative_goal,training=True)
				mu = mu.detach().numpy().squeeze()
				sigma = torch.sqrt(torch.exp(logvar)).detach().numpy().squeeze()

				policies[i_x,i_y,:] = mu

	return policies 


def make_state_space(param):

	dx = 0.025
	robot_idx = 0 
	X = np.arange(param.env_xlim[0],param.env_xlim[1],dx)
	Y = np.arange(param.env_ylim[0],param.env_ylim[1],dx)

	df_state = np.array(copy.copy(param.state)) # num_robots x state_dim 

	states = np.zeros((X.shape[0],Y.shape[0],df_state.shape[0],df_state.shape[1]))
	for i_x,x in enumerate(X): 
		for i_y,y in enumerate(Y): 
			state = df_state 
			state[robot_idx,0] = x 
			state[robot_idx,1] = y 
			states[i_x,i_y,:,:] = state

	return X,Y,states


def make_trees(param):

	from cpp_interface import self_play

	policy_dict_1 = {
		'sim_mode' : 				"D_MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
		'path_glas_model_a' : 		"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/a12.pt",
		'path_glas_model_b' : 		"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/b12.pt",
		'path_value_fnc' : 			"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/v12.pt",
		'mcts_tree_size' : 			500,
		'mcts_c_param' : 			2.0,
		'mcts_pw_C' : 				1.0,
		'mcts_pw_alpha' : 			0.25,
		'mcts_beta1' : 				0.0,
		'mcts_beta2' : 				0.5,
		'mcts_beta3' : 				0.5,
	}

	policy_dict_2 = {
		'sim_mode' : 				"MCTS", # "MCTS, D_MCTS, RANDOM, PANAGOU, GLAS"
		'path_glas_model_a' : 		"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/a12.pt",
		'path_glas_model_b' : 		"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/b12.pt",
		'path_value_fnc' : 			"/home/ben/projects/swarm_ai/saved/double_integrator_hardware/v12.pt",
		'mcts_tree_size' : 			10000,
		'mcts_c_param' : 			2.0,
		'mcts_pw_C' : 				1.0,
		'mcts_pw_alpha' : 			0.25,
		'mcts_beta1' : 				0.0,
		'mcts_beta2' : 				0.5,
		'mcts_beta3' : 				0.5,
	}	

	param = copy.copy(param)

	param.sim_dt = 0.25
	param.robot_types["standard_robot"]["tag_radius"] = 0.02
	param.robot_types["standard_robot"]["goal_radius"] = 0.02
	param.env_l = 2.0
	param.plot_tree_on = True
	# param.update()

	sim_results = [] 

	param.policy_dict = policy_dict_1
	sim_results.append(self_play(param))

	param.policy_dict = policy_dict_2 
	sim_results.append(self_play(param))

	return sim_results 

if __name__ == '__main__':

	param = Param() 
	# param.init_on_sides = False 
	param.update()

	# 0 : plot trees 
	# 1 : plot policy and values 

	modes = [0,1] 

	if 0 in modes: 
		sim_results = make_trees(param)
		for sim_result in sim_results:
			plotter.plot_exp10_trees(param,sim_result)

	if 1 in modes: 

		X,Y,S = make_state_space(param)

		P = evaluate_policy(param,S)
		V = evaluate_values(param,S)

		plotter.plot_exp10_policies(param,X,Y,S,P)
		plotter.plot_exp10_values(param,X,Y,S,V)

		
	print('saving and opening figs...')
	plotter.save_figs('plots/exp10.pdf')
	plotter.open_figs('plots/exp10.pdf')


