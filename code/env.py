

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
from node import Node, Dynamics, Measurements

# Swarm Testbed environment 
class Swarm(Env):

	def __init__(self, param):
		self.param = param 


	def step(self, actions):

		for node in self.nodes: 
			action_i = actions[node]
			node.forward(action_i)

		self.state = self.nodes_to_state_vec()

		# outputs
		s = self.state
		d = self.done()
		r = self.reward()
		info_dict = self.append_info_dict()

		self.timestep += 1
		return s, r, d, info_dict 


	def observe(self):

		observations = dict()

		# get measurements and put in message 
		for node in self.nodes:
			measurement_i = node.measure(self.state)
			observations[node] = measurement_i
		
		return observations


	def get_reset(self):

		reset = dict()

		# init positions 
		nodes = [] 
		state_dim = 0 
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

			nodes.append(node) 
			state_dim += np.shape(node["state"])[0]

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
			state_estimate = state_initial + \
				np.dot(state_covariance,np.random.normal(size=((state_dim,1))))

			node["state_estimate"] = state_estimate
			node["state_covariance"] = state_covariance 

		reset["nodes"] = nodes 
		reset["state_initial"] = state_initial
		reset["state_dim"] = state_dim

		return reset 


	def reset(self,reset):
		# input:
		# 	- 
		# output:
		# 	- 
		
		self.state = reset["state_initial"] 
		self.param.state_dim = reset["state_dim"] 
		self.timestep = 0 
		
		self.nodes = []
		self.nodes_A = []
		self.nodes_B = [] 

		# create nodes 
		for node_dict in reset["nodes"]:
			node = Node(node_dict)

			# assign a system 
			node.dynamics = Dynamics(self.param)
			node.measurements = Measurements(self.param)

			self.nodes.append(node)

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