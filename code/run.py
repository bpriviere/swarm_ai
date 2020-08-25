

# standard packages
import numpy as np 
import time as time_pkg
import os , sys
import glob
import time 
import shutil
import itertools
import multiprocessing as mp 

# my packages 
from param import Param
from env import Swarm
from utilities import load_module, dbgp
import datahandler
import plotter 

def format_dir(param):

	# clean current results dir 
	for old_sim_result_dir in glob.glob(param.current_results_dir):
		shutil.rmtree(old_sim_result_dir)

	# make current results dir 
	os.makedirs(param.current_results_dir,exist_ok=True)


def sim(param):

	env = Swarm(param)
	estimator = load_module(param.estimator_name).Estimator(param,env)
	attacker = load_module(param.attacker_name).Attacker(param,env)
	controller = load_module(param.controller_name).Controller(param,env)
	reset = env.get_reset()

	sim_result = run_sim(param,env,reset,estimator,attacker,controller)

	# write sim results
	results_fn = param.current_results_dir + '/sim_result_{}'.format(param.trial)
	datahandler.write_sim_result(sim_result,results_fn)


def prepare_run(df_param):
	
	params = [] 

	for trial in range(df_param.num_trials):
		
		param = Param()
		param.seed = int.from_bytes(os.urandom(4), sys.byteorder)
		param.trial = trial
		# param.update(df_param.state)
		param.update()
		params.append(param)

	return params 



def run_sim(param,env,reset,estimator,attacker,controller):

	sim_result = dict() 
	times, states, actions, observations, rewards = [],[],[],[],[]

	env.reset(reset)
	states.append(env.state_dict)
	# action = controller.initial_policy()
	estimate = estimator.initial_estimate()

	# for time in tqdm(param.sim_times):
	for step,time in enumerate(param.sim_times):
		if not param.quiet_on: 
			print('\t\t t = {}/{}'.format(step,len(param.sim_times)))
		
		observation = env.observe() 
		observation = attacker.attack(observation)
		action = controller.policy(estimate) 
		estimate = estimator.estimate(observation,action)  
		state, reward, done, info = env.step(estimate,action) 

		times.append(time)
		states.append(state)
		actions.append(action)
		observations.append(observation)
		rewards.append(reward)

		if env.is_terminal(): 
			break 

	# states = states[0:-1]
	states = states[1:]

	sim_result["times"] = times 
	sim_result["states"] = save_lst_of_node_dicts_as_np(states)
	sim_result["actions"] = save_lst_of_node_dicts_as_np(actions)
	sim_result["observations"] = observations 
	sim_result["rewards"] = np.array(rewards) 
	sim_result["info"] = info
	sim_result["param"] = param.to_dict() 

	return sim_result


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

	# Load run parameters
	df_param = Param() 

	# prep run directory
	format_dir(df_param)

	# 
	params = prepare_run(df_param)
		
	# run sim 
	if df_param.parallel_on: 
		pool = mp.Pool(np.min((mp.cpu_count()-1,df_param.num_trials)))
		for _ in pool.imap_unordered(sim, params):
			pass 
	else: 
		sim(params[0])

	# load sim results 
	print('loading sim results...')
	sim_results = [] # lst of dicts
	for sim_result_dir in glob.glob(df_param.current_results_dir + '/*'):
		sim_results.append(datahandler.load_sim_result(sim_result_dir))

	# plotting 
	print('plotting sim results...')
	for sim_result in sim_results:
		plotter.plot_tree_results(sim_result)

		if df_param.gif_on: 
			plotter.make_gif(sim_result)

	print('saving and opening figs...')
	plotter.save_figs(df_param.plot_fn)
	plotter.open_figs(df_param.plot_fn)
