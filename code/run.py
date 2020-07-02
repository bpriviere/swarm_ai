
# standard packages
import numpy as np 
import time as time_pkg
import os 
import glob
import shutil
import itertools

# my packages 
from param import Param
from env import Swarm
from utilities import load_module, dbgp
import datahandler
import plotter 

def format_dir(param):

	# make current results dir 
	if not os.path.exists(param.current_results_dir + '/*'):
		os.makedirs(param.current_results_dir + '/*',exist_ok=True)

	# clean current results dir 
	for old_sim_result_dir in glob.glob(param.current_results_dir):
		shutil.rmtree(old_sim_result_dir)


def sim(param):

	env = Swarm(param)

	estimator = load_module('estimator/exact.py').Estimator(param,env)
	attacker = load_module('attacker/empty.py').Attacker(param,env)
	controller = load_module('controller/joint_mpc.py').Controller(param,env)

	reset = env.get_reset()

	sim_result = dict() 
	times, states, actions, observations, rewards = [],[],[],[],[]

	env.reset(reset)
	states.append(env.state)
	action = controller.initial_policy()

	for step,time in enumerate(param.sim_times):
		print('\t\t t = {}/{}'.format(step,len(param.sim_times)))
		
		observation = env.observe() 
		observation = attacker.attack(observation)
		estimate = estimator.estimate(observation,action)  
		action = controller.policy(estimate) 
		state, reward, done, info = env.step(estimate,action) 

		times.append(time)
		states.append(state)
		actions.append(action)
		observations.append(observation)
		rewards.append(reward)

	states = states[0:-1]

	sim_result["times"] = times 
	sim_result["states"] = states 
	sim_result["actions"] = save_lst_of_node_dicts_as_np(actions)
	sim_result["observations"] = observations 
	sim_result["rewards"] = rewards 
	sim_result["info"] = info
	sim_result["param"] = param.to_dict() 

	return [sim_result]


def save_lst_of_node_dicts_as_np(some_lst_of_node_dicts):

	# array = [nt x ni x dim_value] 
	num_timesteps = len(some_lst_of_node_dicts)
	num_agents = len(some_lst_of_node_dicts[0])

	for node, value in some_lst_of_node_dicts[0].items():
		value_shape = value.shape
		break

	shape = (num_timesteps,num_agents) + value_shape
	array = np.zeros(shape)

	for timestep, some_dict in enumerate(some_lst_of_node_dicts):
		for node, value in some_dict.items():
			array[timestep,node.idx,:,:] = value

	return array


if __name__ == '__main__':

	param = Param() 

	# prep run directory
	format_dir(param)
		
	# run sim 
	sim_results = sim(param)

	# write sim results
	print('writing sim results...')
	for case_i,sim_result in enumerate(sim_results):
		results_dir = param.current_results_dir + '/sim_result_{}'.format(case_i)
		datahandler.write_sim_result(sim_result,results_dir)

	# load sim results 
	print('loading sim results...')
	sim_results = [] # lst of dicts
	for sim_result_dir in glob.glob(param.current_results_dir + '/*'):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	# plotting 
	print('plotting sim results...')
	if True: #param.n_trials == 1:
		for sim_result in sim_results:

			for timestep,time in enumerate(sim_result["times"]):
				plotter.plot_nodes(sim_result,timestep) # at one timestep 
			# plotter.plot_state_estimate(sim_result) # over time 
			plotter.plot_control_effort(sim_result)
			plotter.plot_speeds(sim_result)

	print('saving and opening figs...')
	plotter.save_figs(param.plot_fn)
	plotter.open_figs(param.plot_fn)
