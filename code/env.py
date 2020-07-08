

# standard package
import numpy as np 
import copy
from gym import Env
from numpy.random import random
from matplotlib import pyplot as plt 
import importlib

# my package
import plotter 
from utilities import dbgp, load_module

class Swarm(Env):

	def __init__(self, param):
		self.param = param 
		self.reset(self.get_reset())


	def step(self, estimates, actions):

		# update estimates
		for node in self.nodes:
			estimates[node].update_node(node) 

		# regular dynamics 
		for node in self.nodes: 
			action_i = actions[node]
			node.forward(action_i)

		# check for tag
		for node_i in self.nodes:  
			for node_j in self.nodes: 
				if node_i.dist_to_node(node_j) < self.param.tag_radius and \
					node_i.team_A and node_j.team_B:
					
					node_i.reset_inside(self.param.reset_xlim_A,self.param.reset_ylim_A,self.state_vec)
					node_j.reset_inside(self.param.reset_xlim_B,self.param.reset_ylim_B,self.state_vec)

		self.state_vec = self.nodes_to_state_vec()
		self.state_dict = dict()
		for node in self.nodes: 
			self.state_dict[node] = node.state

		# outputs
		s = self.state_dict
		d = self.done()
		r = self.reward()
		info_dict = self.append_info_dict()

		self.timestep += 1
		return s, r, d, info_dict 


	def observe(self):

		observations = dict()

		# get measurements and put in message 
		for node in self.nodes:
			measurement_i = node.measure(self.state_vec)
			observations[node] = measurement_i
		
		return observations


	def get_reset(self):

		reset = dict()

		# init positions 
		nodes = [] 
		state_dim = 0 
		control_dim = 0 
		for idx in range(self.param.num_nodes):

			node = dict()

			if idx < self.param.num_nodes_A:
				node["team_A"] = True
				node["team_B"] = False
				xlim = self.param.reset_xlim_A
				ylim = self.param.reset_ylim_A
			else: 
				node["team_A"] = False
				node["team_B"] = True
				xlim = self.param.reset_xlim_B
				ylim = self.param.reset_ylim_B

			position = self.get_random_position_inside(xlim,ylim)
			node["state"] = np.array([
				[position[0]],
				[position[1]],
				[0.0],
				[0.0]])
			node["idx"] = idx
			node["global_state_idxs"] = state_dim + np.arange(4)
			node["global_control_idxs"] = control_dim + np.arange(2)

			nodes.append(node) 
			state_dim += np.shape(node["state"])[0]
			control_dim += 2 

		# get state
		state_initial = np.zeros((state_dim,1))
		curr_idx = 0
		for node in nodes:
			state_idx = curr_idx + np.arange(np.shape(node["state"])[0])
			state_initial[state_idx] = node["state"]
			curr_idx += np.shape(node["state"])[0]

		# init state estimates 
		for node in nodes: 

			state_covariance = self.param.initial_state_covariance*np.eye(state_dim)
			state_mean = state_initial + \
				np.dot(state_covariance,np.random.normal(size=((state_dim,1))))

			node["state_mean"] = state_mean
			node["state_covariance"] = state_covariance 

		reset["nodes"] = nodes 
		reset["state_initial"] = state_initial
		reset["state_dim"] = state_dim
		reset["control_dim"] = control_dim 

		return reset 


	def reset(self,reset):
		# input:
		# 	- 
		# output:
		# 	- 
		
		self.state_vec = reset["state_initial"] 
		self.param.state_dim = reset["state_dim"] 
		self.param.control_dim = reset["control_dim"] 
		self.timestep = 0 
		
		self.nodes = []
		self.nodes_A = []
		self.nodes_B = [] 

		self.state_dict = dict()

		# create nodes 
		for node_dict in reset["nodes"]:
			node = Node(self.param,node_dict)

			# assign a system 
			node.dynamics = load_module(self.param.dynamics_name).Dynamics(self.param)
			node.measurements = load_module(self.param.measurements_name).Measurements(self.param)

			self.nodes.append(node)
			self.state_dict[node] = node.state

		self.info_dict = self.init_info_dict()


	def append_info_dict(self):

		for key in self.param.info_keys:
			if 'node_' in key:
				node_key = key.split('node_')[-1]
				value = [] 
				for node in self.nodes:
					value.append(node.__dict__[node_key])
			else:
				value = self.__dict__[key]
			self.info_dict[key].append(value)
		return self.info_dict


	def init_info_dict(self):

		info_dict = dict()
		for key in self.param.info_keys:
			if 'node_' in key:
				node_key = key.split('node_')[-1]
				value = [] 
				for node in self.nodes:
					value.append(node.__dict__[node_key])
			else:
				value = self.__dict__[key]
			info_dict[key] = []

		# state is special
		if "state" in self.param.info_keys:
			info_dict["state"] = [self.state]

		return info_dict		


	def nodes_to_state_vec(self):
		
		state = np.zeros((self.param.state_dim,1))
		curr_state_dim = 0
		for node in self.nodes: 
			
			state_idxs = curr_state_dim + np.arange(0,node.dynamics.state_dim_per_agent)
			curr_state_dim += node.dynamics.state_dim_per_agent
			state[state_idxs] = node.state

		return state 


	def state_vec_to_dict(self,state_vec):

		states = dict()
		for node in self.nodes: 
			states[node] = state_vec[node.global_state_idxs]

		return states


	def get_random_position_inside(self,xlim,ylim):

		x = np.random.random()*(xlim[1] - xlim[0]) + xlim[0]
		y = np.random.random()*(ylim[1] - ylim[0]) + ylim[0]
		
		return x,y 				


	def done(self):
		return False


	def render(self):
		return 0 


	def check_feasible_instance(self):
		return True
		

	def reward(self):
		return 0 


# helper classes 
class Node:

	def __init__(self,param,node_dict):
		self.param = param 
		for key,value in node_dict.items():
			setattr(self,key,value)

	def dist_to_node(self,node):
		return np.linalg.norm(self.state[0:2]-node.state[0:2])

	def measure(self,full_state):
		return self.measurements.measure_per_node(full_state)

	def forward(self,control):
		self.state = self.dynamics.forward_per_node(self.state,control)

	def reset_inside(self,xlim,ylim,state):

		x = np.random.random()*(xlim[1] - xlim[0]) + xlim[0]
		y = np.random.random()*(ylim[1] - ylim[0]) + ylim[0]

		# self.state = np.expand_dims(np.array([
		# 	x,y,0.0,0.0]),axis=1)
		self.state = np.array([
			[x],[y],[0.0],[0.0]])

		self.state_covariance = self.param.initial_state_covariance*np.eye(self.param.state_dim)
		self.state_mean = state + \
				np.dot(self.state_covariance,np.random.normal(size=((self.param.state_dim,1))))