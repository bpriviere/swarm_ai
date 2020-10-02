

import os, sys
import yaml  
import torch 
import numpy as np 
from multiprocessing import current_process
import tqdm 

sys.path.append('../')
import datahandler as dh 
from learning.continuous_emptynet import ContinuousEmptyNet
from learning_interface import format_data, global_to_local

def test_evaluate_expert(states,param,testing,quiet_on=True,progress=None):
	from cpp_interface import param_to_cpp_game, is_valid_policy_dict, valuePerAction_to_policy_dist
	
	alphas = np.random.randint(low=0,high=len(testing),size=len(states))
	
	if not quiet_on:
		print('   running expert for instance {}'.format(param.dataset_fn))

	if progress is not None:
		progress_pos = current_process()._identity[0] - progress
		# enumeration = tqdm.tqdm(states, desc=param.dataset_fn, leave=False, position=progress_pos)
		enumeration = tqdm.tqdm(zip(alphas,states), desc=param.dataset_fn, leave=False, position=progress_pos)
	else:
		enumeration = zip(alphas,states)

	if is_valid_policy_dict(param.policy_dict):
		policy_dict = param.policy_dict
	else: 
		print('bad policy dict')
		exit()

	g = param_to_cpp_game(param.robot_team_composition,param.robot_types,param.env_xlim,param.env_ylim,\
		param.sim_dt,param.goal,param.rollout_horizon)	

	sim_result = {
		'states' : [],
		'policy_dists' : [],
		'values' : [],
		'param' : param.to_dict()
		}

	for alpha, state in enumeration:

		test_state = np.array(testing[alpha]["test_state"])
		test_value = testing[alpha]["test_value"]

		test_valuePerAction = []
		for valuePerAction in testing[alpha]["test_valuePerAction"]:
			x = ((np.array((valuePerAction[0],valuePerAction[1]),dtype=float),np.array((np.nan,np.nan),dtype=float)),valuePerAction[-1])
			test_valuePerAction.append(x)

		test_policy_dist = valuePerAction_to_policy_dist(param,test_valuePerAction) # 

		sim_result["states"].append(test_state)
		sim_result["policy_dists"].append(test_policy_dist)  		
		sim_result["values"].append(test_value)

	sim_result["states"] = np.array(sim_result["states"])
	sim_result["policy_dists"] = np.array(sim_result["policy_dists"])
	dh.write_sim_result(sim_result,param.dataset_fn)

	if not quiet_on:
		print('   completed instance {} with {} dp.'.format(param.dataset_fn,sim_result["states"].shape[0]))	

def test_evaluate_expert_wrapper(arg):
	# When using multiprocessing, load cpp_interface per process
	# from cpp_interface import test_evaluate_expert
	test_evaluate_expert(*arg)

def test_model(param,model_fn,testing):

	print('testing model: {}'.format(model_fn))

	stats = {} 
	model = ContinuousEmptyNet(param, "cpu")
	model.load_state_dict(torch.load(model_fn))

	robot_idx = 0 
	n_samples = 1000

	for alpha in range(len(testing)):
		
		stats_per_condition = {
			'learned' : [],
			'test' : [],
			'latent' : [], 
		}

		test_state = np.array(testing[alpha]["test_state"])
		o_a,o_b,goal = global_to_local(test_state,param,robot_idx)
		o_a,o_b,goal = format_data(o_a,o_b,goal)

		value, _ = model(o_a,o_b,goal)

		print('test num: ', alpha)
		print('   value: ',value)
		print('   valuePerAction: ', testing[alpha]["test_valuePerAction"])

		# learned distribution 
		for i_sample in range(n_samples):
			_, learned_sample = model(o_a,o_b,goal)
			stats_per_condition["learned"].append(learned_sample.detach().numpy())

		# test distribution 
		weights = []
		actions = [] 
		for action in testing[alpha]["test_valuePerAction"]:
			weights.append(action[-1])
			actions.append(np.array((action[0],action[1])))

		weights = np.array(weights)
		actions = np.array(actions)
		weights /= sum(weights) 
		choice_idxs = np.random.choice(actions.shape[0],n_samples,p=weights)
		for choice_idx in choice_idxs:
			mu = actions[choice_idx,:]
			test_sample = mu + 0.01 * np.random.normal(size=(2,)) 
			stats_per_condition["test"].append(test_sample)

		stats_per_condition["learned"] = np.array(stats_per_condition["learned"]).squeeze()
		stats_per_condition["test"] = np.array(stats_per_condition["test"]).squeeze()

		stats[alpha] = stats_per_condition

	return stats

def read_testing_yaml(fn):
	
	with open(fn) as f:
		testing_cfg = yaml.load(f, Loader=yaml.FullLoader)

	testing = []
	for test in testing_cfg["test_continuous_glas"]:
		testing.append(test)
	return testing

