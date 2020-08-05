
import numpy as np
import os
import shutil
import json
import glob
import yaml
import pandas as pd
import pickle
from datetime import datetime,timedelta

from utilities import dbgp

def write_sim_result(sim_result_dict,fn):
	with open(fn+'.pickle', 'wb') as h:
		pickle.dump(sim_result_dict, h)

def load_sim_result(fn):
	with open(fn, 'rb') as h:
		sim_result = pickle.load(h)
	return sim_result
	
def write_state_action_pairs(sim_result,fn):

	states = np.array(sim_result["info"]["state_vec"]) # [nt x state_dim x 1]
	actions = sim_result["actions"] # [nt x ni x action_dim_per_agent x 1]
	times = np.expand_dims(sim_result["times"],axis=1) # [nt x 1]

	num_timesteps, num_agents, action_dim_per_agent, _ = actions.shape
	_, state_dim, _ = states.shape

	flat_actions = np.zeros((num_timesteps, num_agents*action_dim_per_agent,1))
	for timestep,action in enumerate(actions): 
		new_idxs = np.arange(action_dim_per_agent)
		for node_idx, action_i in enumerate(action): 
			flat_actions[timestep,new_idxs,:] = action_i
			new_idxs += action_dim_per_agent

	state_action_pairs = np.zeros((num_timesteps, state_dim + num_agents*action_dim_per_agent))
	for timestep,(state,action) in enumerate(zip(states,flat_actions)):
		state_action_pairs[timestep,:] = np.vstack((state,action)).flatten()

	np.save(fn,state_action_pairs)


def write_mcts_config_file(param, config_fn):

	config = param.to_dict()
	for key, value in config.items():
		if isinstance(value,np.ndarray):
			config[key] = value.tolist()

	with open(config_fn,'w') as f:
		yaml.dump(config,f)

def write_mcts_state_action_pairs(data, fn, param):

	key = os.path.basename(fn)
	key = key.split('_')
	num_a = int(key[1].split('a')[0])
	num_b = int(key[2].split('b')[0])
	states = data[0:-1,:]

	actions = np.zeros((data.shape[0]-1,2*(num_a + num_b)))
	for node_idx in range(num_a + num_b):
		action_idxs = node_idx * 2 + np.arange(2)
		data_idxs = node_idx * 4 + np.arange(2) + 2 
		# actions[:,action_idxs] = (data[1:,data_idxs] - data[0:-1,data_idxs]) / param.sim_dt

		pos_x = np.where(data[1:,data_idxs[0]] > data[0:-1,data_idxs[0]])
		neg_x = np.where(data[1:,data_idxs[0]] < data[0:-1,data_idxs[0]])
		equ_x = np.where(data[1:,data_idxs[0]] == data[0:-1,data_idxs[0]])

		pos_y = np.where(data[1:,data_idxs[1]] > data[0:-1,data_idxs[1]])
		neg_y = np.where(data[1:,data_idxs[1]] < data[0:-1,data_idxs[1]])
		equ_y = np.where(data[1:,data_idxs[1]] == data[0:-1,data_idxs[1]])

		actions[pos_x,action_idxs[0]] = 1 
		actions[neg_x,action_idxs[0]] = 0 
		actions[equ_x,action_idxs[0]] = -1 
		
		actions[pos_y,action_idxs[1]] = 1 
		actions[neg_y,action_idxs[1]] = 0 
		actions[equ_y,action_idxs[1]] = -1 

	state_action_pairs = np.hstack((states,actions))

	np.save(fn,state_action_pairs)


def write_oa_pair_batch(batched_dataset,batch_fn):
	np.save(batch_fn, batched_dataset)


def write_parameters(param_dict,fn):

	with open(fn, 'w') as fp:
		json.dump(param_dict, fp, cls=NumpyEncoder, indent=2)


def read_parameters(fn):

	with open(fn, 'r') as j:
		param_dict = json.loads(j.read())
	return param_dict


def read_state_action_pairs(fn,param):

	data = np.load(fn)
	states = data[:,0:param.state_dim]
	actions = data[:,param.state_dim:]
	return states,actions


def read_observation_action_pairs(fn,datadir):

	data = np.load(fn)

	key = fn.split(datadir)[-1]
	key = key.split('_')
	num_a = int(key[2].split('a')[0])
	num_b = int(key[3].split('b')[0])
	action_dim_per_agent = 2
	state_dim_per_agent = 4

	o_a = data[:,0:num_a*state_dim_per_agent]
	o_b = data[:,num_a*state_dim_per_agent:(num_a+num_b)*state_dim_per_agent]
	goal = data[:,(num_a+num_b)*state_dim_per_agent:(num_a+num_b)*state_dim_per_agent+state_dim_per_agent]
	action = data[:,(num_a+num_b)*state_dim_per_agent+state_dim_per_agent:]
	
	return o_a,o_b,goal,action 

