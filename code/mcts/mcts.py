
import numpy as np 
from collections import defaultdict
import itertools
import random 
import copy

import sys
sys.path.append("../")
from utilities import dbgp


def sample(actions):
	return actions[np.random.randint(len(actions))]

class State:

	def __init__(self,param,state,done,turn):
		self.param = param 
		self.state = state
		self.turn = turn 
		self.done = done 
		self.not_done = list(set(self.param.team_1_idxs)-set(self.done))

		if self.turn: 
			self.u_max = self.param.acceleration_limit_a
			self.v_max = self.param.speed_limit_a
			self.idxs = self.not_done 
		else:
			self.u_max = self.param.acceleration_limit_b
			self.v_max = self.param.speed_limit_b
			self.idxs = self.param.team_2_idxs

		self.A = np.array([
			[1,0,self.param.sim_dt,0],
			[0,1,0,self.param.sim_dt],
			[0,0,1,0],
			[0,0,0,1]
			])
		self.B = np.array([
			[0,0],
			[0,0],
			[self.param.sim_dt,0],
			[0,self.param.sim_dt]])

		self.dist_robots = self.make_dist_robots()
		self.dist_goal = self.make_dist_goal()

	def __repr__(self):
		return "State(state={}, turn={}, done={})".format(self.state, self.turn, self.done)

	def make_dist_robots(self):
		dist_robots = np.linalg.norm(self.state[:,0:2][:, np.newaxis] - self.state[:,0:2],axis=2)
		return dist_robots 

	def make_dist_goal(self):
		idxs = self.param.team_1_idxs
		dist_goal = np.zeros(len(idxs))
		dist_goal[:] = np.linalg.norm(self.state[idxs,0:2] - self.param.goal)
		return dist_goal

	def forward(self,action):
		next_state = np.copy(self.state)
		for idx in self.idxs:
			next_state[idx,:] = np.dot(self.A,self.state[idx,:]) + np.dot(self.B,action[idx,:])

		next_done = []
		for idx in self.param.team_1_idxs:
			captured = np.any(self.dist_robots[idx,self.param.team_2_idxs] < self.param.tag_radius)
			reached_goal = self.dist_goal[idx] < self.param.tag_radius
			
			if idx in self.done or captured: 
				next_done.append(idx)
			if reached_goal: 
				next_done = self.param.team_1_idxs
				break 

		next_turn = not self.turn 
		return State(self.param,next_state,next_done,next_turn)

	def is_terminal(self):
		return len(self.done) == len(self.param.team_1_idxs) or len(self.get_legal_actions()) == 0

	def is_valid(self):
		for idx in self.idxs: 
			if not (self.state[idx,0] > self.param.env_xlim[0] and \
				self.state[idx,0] < self.param.env_xlim[1] and \
				self.state[idx,1] > self.param.env_ylim[0] and \
				self.state[idx,1] < self.param.env_ylim[1] and \
				np.linalg.norm(self.state[idx,2:]) < self.v_max): 
				return False 
		return True 

	def eval_reward(self):
		if np.any(self.dist_goal[self.param.team_1_idxs] < self.param.tag_radius):
			reward_1 = 1 
			reward_2 = 0
		elif len(self.done) == len(self.param.team_1_idxs):
			reward_1 = 0 
			reward_2 = 1
		else: 
			reward_1,reward_2 = self.eval_predict()
		return reward_1,reward_2

	def eval_predict(self):
		reward_1,reward_2 = 0,0
		for idx in self.param.team_1_idxs: 
			reward_1 += np.exp(-1*self.dist_goal[idx])
			reward_2 += np.exp(-1*np.sum(self.dist_robots[idx,self.param.team_2_idxs]))
		reward_1 /= len(self.param.team_1_idxs)
		reward_2 /= len(self.param.team_1_idxs)
		return reward_1,reward_2

	def get_legal_actions(self):
		u_xs = self.u_max*np.asarray([-1,0,1])
		u_ys = self.u_max*np.asarray([-1,0,1])

		master_lst = [] 
		for idx in self.idxs:
			master_lst.extend([u_xs,u_ys])
		master_lst = list(itertools.product(*master_lst))

		actions = []
		for elem in list(master_lst):
			action = np.zeros((self.state.shape[0],2))
			for action_idx,robot_idx in enumerate(self.idxs): 
				action[robot_idx,:] = np.array(elem)[action_idx*2 + np.arange(2)] 
			if self.forward(action).is_valid(): 
				actions.append(action)
		return actions


class Node: 
	
	def __init__(self,param,state,parent=None,action_to_node=None):
		self.param = param
		self.state = state 
		self.parent = parent 
		self.action_to_node = action_to_node 

		self.number_of_visits = 0.
		self.value_1 = 0.
		self.value_2 = 0.
		self.children = []
		self.children_weights = []

		self.is_terminal_node = self.state.is_terminal()
		self.untried_actions = self.state.get_legal_actions()
		random.shuffle(self.untried_actions)

	def __repr__(self):
		children_list = [(w, c.action_to_node) for w, c in zip(self.children_weights, self.children)]
		children_list.sort(key = lambda x: x[0], reverse=True)
		return "Node(state={}, n={}, v1={}, v2={}, children={})".format(self.state, self.number_of_visits, self.value_1, self.value_2, children_list)

	def q(self, team_1_turn):
		value = self.value_1 if team_1_turn else self.value_2 
		return value

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def expand(self):
		action = self.untried_actions.pop()
		next_state = self.state.forward(action)
		child_node = Node(self.param,next_state,parent=self,action_to_node=action)
		self.children.append(child_node)
		self.children_weights.append(None)
		return child_node

	def best_child(self, c_param=1.4):
		for k, c in enumerate(self.children): 
			self.children_weights[k] = (c.q(self.state.turn) / c.number_of_visits) + \
				c_param * np.sqrt((2 * np.log(self.number_of_visits) / c.number_of_visits))
		return self.children[np.argmax(self.children_weights)]


class Tree: 

	def __init__(self,param):
		self.param = param 
		self.num_nodes = 0 
		self.root_node = None

	def __repr__(self):
		return "Tree(root_node={}, num_nodes={})".format(self.root_node, self.num_nodes)

	def grow(self):	
		for _ in range(self.param.tree_size):
			current_node = self.tree_policy()
			reward_1,reward_2 = self.rollout(current_node)
			self.backpropagate(current_node,reward_1,reward_2)

	def set_root(self,root_state):
		root_node = Node(self.param,root_state)
		if self.root_node is None: 
			self.num_nodes = 1
		else: 
			self.remove_branch(self.root_node,root_node)
		self.root_node = root_node

	def tree_policy(self):
		current_node = self.root_node
		while not current_node.is_terminal_node:
			if not current_node.is_fully_expanded():
				return current_node.expand()
			else:
				current_node = current_node.best_child()
		return current_node

	def best_action(self):
		if self.root_node.is_terminal_node:
			best_child = self.root_node 
		else: 
			best_child = self.root_node.best_child(c_param=0.0)
		return best_child.state, best_child.action_to_node

	def backpropagate(self,node,reward_1,reward_2):
		node.number_of_visits += 1.
		node.value_1 += reward_1
		node.value_2 += reward_2
		if node.parent:
			self.backpropagate(node.parent,reward_1,reward_2)

	def rollout(self,node):
		state = node.state
		reward_1,reward_2 = state.eval_reward()
		for i in range(self.param.rollout_horizon):
			if state.is_terminal(): 
				return reward_1,reward_2
			actions = state.get_legal_actions()
			action = sample(actions)
			state = state.forward(action)
			reward_1,reward_2 = state.eval_reward()
		return reward_1,reward_2