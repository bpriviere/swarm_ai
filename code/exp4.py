
# standard 
import numpy as np 
import os, glob
import copy
import multiprocessing as mp 
import torch 

# custom 
from param import Param 
from learning.discrete_emptynet import DiscreteEmptyNet
from mice import format_data, relative_state
from cpp_interface import self_play, expected_value
import plotter 
import datahandler as dh

def eval_value(param):
	print('{}/{}'.format(param.count,param.total))

	if param.sim_mode == "GLAS":

		state = np.array(param.state)
		robot_idx = 0 

		o_a,o_b,goal = relative_state(state,param,robot_idx)
		o_a,o_b,goal = format_data(o_a,o_b,goal)

		model = DiscreteEmptyNet(param, "cpu")
		model.load_state_dict(torch.load(param.path_glas_model_a))

		value,action = model(o_a,o_b,goal)

		sim_result = {
			"states" : np.array([param.state]),
			"rewards" : np.array([[value,1-value]]),
			"param" : param.to_dict(),
		}

	elif "MCTS" in param.sim_mode:
		sim_result = self_play(param)

	elif param.sim_mode == "EXPECTED_VALUE": 
		value = expected_value(param) # query tree 
		sim_result = {
			"states" : np.array([param.state]),
			"rewards" : np.array([[value[0],value[1]]]),
			"param" : param.to_dict(),
		}

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
	total = df_param.num_trials * len(df_param.sim_modes) * len(df_param.dss)
	for i_trial in range(df_param.num_trials):
		for sim_mode in df_param.sim_modes: 
			for pos in df_param.dss: 
				initial_condition = copy.deepcopy(df_param.state)
				initial_condition[0][0:2] = pos 

				param = Param() 
				param.X = df_param.X
				param.Y = df_param.Y
				param.num_trials = df_param.num_trials
				param.total = total
				param.mcts_tree_size = df_param.mcts_tree_size
				param.sim_modes = df_param.sim_modes
				param.path_glas_model_a = df_param.path_glas_model_a
				param.path_glas_model_b = df_param.path_glas_model_b		
				
				param.i_trial = i_trial 
				param.sim_mode = sim_mode
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
		df_param.sim_modes = ["EXPECTED_VALUE","GLAS"] #["GLAS"]
		df_param.path_glas_model_a = '../saved/value_fnc_test/a3.pt'
		df_param.path_glas_model_b = '../saved/value_fnc_test/b3.pt'
		df_param.mcts_tree_size = 10000
		dx = 0.05
		df_param.dss, df_param.X, df_param.Y = discretize_state_space(df_param,dx,dx)
		pos = {
			1 : np.array((0.35,0.3))
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
	# 	if not sim_result["param"]["sim_mode"] == "EXPECTED_VALUE":
	# 		plotter.plot_tree_results(sim_result,title=sim_result["param"]["sim_mode"])
	# 		count += 1 
	# 	if count > 10:
	# 		break 
	
	plotter.save_figs("plots/exp4.pdf")
	plotter.open_figs("plots/exp4.pdf")



if __name__ == '__main__':
	main()