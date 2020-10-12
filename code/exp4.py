
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
from learning_interface import format_data, global_to_local 
import plotter 
import datahandler as dh

def eval_value(param):
	# When using multiprocessing, load cpp_interface per process
	from cpp_interface import self_play, expected_value
	
	print('{}/{}'.format(param.count,param.total))

	if param.exp4_sim_mode == "GLAS_VALUE":

		state = np.array(param.state)
		robot_idx = 0 

		o_a,o_b,goal = global_to_local(state,param,robot_idx)
		o_a,o_b,goal = format_data(o_a,o_b,goal)

		if param.l_gaussian_on: 
			model = GaussianEmptyNet(param,"cpu")
		else: 
			model = ContinuousEmptyNet(param,"cpu")

		model.load_state_dict(torch.load(param.policy_dict["path_glas_model_a"]))

		value,action = model(o_a,o_b,goal)

		sim_result = {
			"states" : np.array([param.state]),
			"rewards" : np.array([[value,1-value]]),
			"param" : param.to_dict(),
		}

	elif param.exp4_sim_mode == "MCTS_VALUE": 
		state = np.array(param.state)
		value = expected_value(param,state,param.policy_dict) # query tree 
		sim_result = {
			"states" : np.array([param.state]),
			"rewards" : np.array([[value[0],value[1]]]),
			"param" : param.to_dict(),
		}

	elif param.exp4_sim_mode == "MCTS_SIM":
		sim_result = self_play(param)

	elif param.exp4_sim_mode == "GLAS_SIM":
		sim_result = self_play(param)

	else: 
		exit('sim_mode not regonzied')

	dh.write_sim_result(sim_result,param.dataset_fn)

def make_initial_condition(df_param,pos):

	initial_condition = [] 
	for i_robot in range(len(df_param.robots)):
		if i_robot == 0:
			pos_x = 0 # temp, will be overwritten
			pos_y = 0 
		elif i_robot in pos.keys():
			pos_x = pos[i_robot][0]
			pos_y = pos[i_robot][1]
		else: 
			print(i_robot)
			exit('initial condition not well specified')

		initial_condition.append([pos_x,pos_y,0,0])
	return initial_condition


def get_params(df_param):

	params = [] 
	count = 0 
	total = df_param.num_trials * len(df_param.exp4_sim_modes) * len(df_param.dss)
	for i_trial in range(df_param.num_trials):
		for exp4_sim_mode in df_param.exp4_sim_modes: 
			for pos in df_param.dss: 
				initial_condition = copy.deepcopy(df_param.state)
				initial_condition[0][0:2] = pos 

				param = Param() 
				param.env_l = df_param.env_l
				param.X = df_param.X
				param.Y = df_param.Y
				param.num_trials = df_param.num_trials
				param.total = total
				param.mcts_tree_size = df_param.mcts_tree_size
				param.sim_modes = df_param.exp4_sim_modes
				param.exp4_sim_mode = exp4_sim_mode

				if 'MCTS' in exp4_sim_mode:
					param.policy_dict["sim_mode"] = "MCTS"
				elif 'GLAS' in exp4_sim_mode: 
					param.policy_dict["sim_mode"] = "GLAS"
				else: 
					exit('exp4 sim_mode not recognized: ', exp4_sim_mode)

				param.policy_dict["path_glas_model_a"] = df_param.path_glas_model_a
				param.policy_dict["path_glas_model_b"] = df_param.path_glas_model_b
				
				param.i_trial = i_trial 
				param.dataset_fn = df_param.path_current_results + 'sim_result_{}'.format(count)
				param.count = count 
				param.update(initial_condition=initial_condition)

				params.append(param)
				count += 1 
	return params 


def discretize_state_space(df_param,dx,dy):

	dss = [] 
	X = np.arange(df_param.env_xlim[0],df_param.env_xlim[1],dx) + dx / 2.0
	Y = np.arange(df_param.env_ylim[0],df_param.env_ylim[1],dy) + dy / 2.0 
	for x in X: 
		for y in Y: 
			dss.append(np.array((x,y)))
	return dss, X, Y


def format_dir(df_param):
	if os.path.exists(df_param.path_current_results):
		for file in glob.glob(df_param.path_current_results + "/*"):
			os.remove(file)
	os.makedirs(df_param.path_current_results,exist_ok=True)


def main():
	run_on = True
	df_param = Param()

	if run_on: 
		df_param.num_trials = 1
		df_param.env_l = 0.5
		df_param.make_environment()
		df_param.exp4_sim_modes = ["GLAS_SIM"] # ["MCTS_VALUE", "MCTS_SIM"] #,"GLAS_VALUE"] 
		df_param.path_glas_model_a = '../current/models/a1.pt'
		df_param.path_glas_model_b = '../current/models/b1.pt'
		df_param.mcts_tree_size = 100000
		dx = 0.05
		df_param.dss, df_param.X, df_param.Y = discretize_state_space(df_param,dx,dx)
		pos = {
			# robot idx : position
			1 : df_param.env_l*np.array((0.35,0.5))
		}
		df_param.state = make_initial_condition(df_param,pos)

		format_dir(df_param)
		params = get_params(df_param)

		if df_param.sim_parallel_on: 	
			pool = mp.Pool(mp.cpu_count()-1)
			for _ in pool.imap_unordered(eval_value, params):
				pass 
		else:
			for param in params: 
				eval_value(param)

	sim_results = [] 
	for sim_result_dir in glob.glob(df_param.path_current_results + '/*'):
		sim_results.append(dh.load_sim_result(sim_result_dir))

	plotter.plot_exp4_results(sim_results)

	# count = 0 
	# for sim_result in sim_results:
	# 	if not sim_result["param"]["sim_mode"] == "MCTS_VALUE":
	# 		plotter.plot_tree_results(sim_result,title=sim_result["param"]["sim_mode"])
	# 		count += 1 
	# 	if count > 10:
	# 		break 
	
	plotter.save_figs("plots/exp4.pdf")
	plotter.open_figs("plots/exp4.pdf")



if __name__ == '__main__':
	main()