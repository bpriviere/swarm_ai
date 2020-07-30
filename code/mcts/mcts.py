
import numpy as np 
from collections import defaultdict
import itertools
import random 
import copy

import sys
sys.path.append("../")
from utilities import dbgp

class RobotDynamics:
	def __init__(self, param, team):
		self.param = param
		self.A = np.array([
			[1,0,param.sim_dt,0],
			[0,1,0,param.sim_dt],
			[0,0,1,0],
			[0,0,0,1]
			])
		self.B = np.array([
			[0,0],
			[0,0],
			[param.sim_dt,0],
			[0,param.sim_dt]])
		self.num_robots = param.num_nodes_A + param.num_nodes_B
		self.possible_actions_dict = dict()

		if team == 'a': 
			self.u_max = param.acceleration_limit_a
			self.v_max = param.speed_limit_a
			self.idxs = param.team_1_idxs
		elif team == 'b':
			self.u_max = param.acceleration_limit_b
			self.v_max = param.speed_limit_b
			self.idxs = param.team_2_idxs

	def step(self, state, action):
		return np.dot(self.A, state) + np.dot(self.B, action)

	def valid(self, state):
		if not (state[0] > self.param.env_xlim[0] and \
				state[0] < self.param.env_xlim[1] and \
				state[1] > self.param.env_ylim[0] and \
				state[1] < self.param.env_ylim[1] and \
				np.linalg.norm(state[2:]) < self.v_max):
			return False
		return True

	def possible_actions(self, idxs):
		s = tuple(sorted(set(idxs)))
		result = self.possible_actions_dict.get(s)
		if result is None:
			result = self.__compute_possible_actions(idxs)
			self.possible_actions_dict[s] = result
		return result

	def __compute_possible_actions(self, idxs):
		u_xs = self.u_max*np.asarray([-1,0,1])
		u_ys = self.u_max*np.asarray([-1,0,1])

		master_lst = [] 
		for idx in self.idxs:
			master_lst.extend([u_xs,u_ys])
		master_lst = list(itertools.product(*master_lst))

		actions = []
		for elem in list(master_lst):
			action = np.zeros((self.num_robots,2))
			for action_idx, robot_idx in enumerate(idxs): 
				action[robot_idx,:] = np.array(elem)[action_idx*2 + np.arange(2)]
			actions.append(action)
		return actions



class State:
	# shared variables for all instances
	param = None
	dynamicsA = None
	dynamicsB = None

	def __init__(self,state,prev_done,turn):
		self.state = state
		self.turn = turn

		self.dist_robots = np.linalg.norm(self.state[:,0:2][:, np.newaxis] - self.state[:,0:2],axis=2)
		self.dist_goal = np.linalg.norm(self.state[self.param.team_1_idxs,0:2] - self.param.goal, axis=1)
		self.done = self.__compute_done(prev_done) 
		if turn:
			self.dynamics = self.dynamicsA
			# the indices are the ones of team 1 that are still actively playing
			self.idxs = list(set(self.param.team_1_idxs)-set(self.done))
		else:
			self.dynamics = self.dynamicsB
			self.idxs = self.param.team_2_idxs

	def __repr__(self):
		return "State(state={}, turn={}, done={})".format(self.state, self.turn, self.done)

	@classmethod
	def init_class(cls, param, dynamicsA, dynamicsB):
		cls.param = param
		cls.dynamicsA = dynamicsA
		cls.dynamicsB = dynamicsB

	def forward(self,action):
		next_state = np.copy(self.state)
		for idx in self.idxs:
			next_state[idx,:] = self.dynamics.step(self.state[idx,:], action[idx,:])
			if not self.dynamics.valid(next_state[idx,:]):
				return None
		next_turn = not self.turn 
		return State(next_state,self.done,next_turn)

	def possible_actions(self):
		return self.dynamics.possible_actions(self.idxs)

	def is_terminal(self):
		return len(self.done) == len(self.param.team_1_idxs)

	def __compute_done(self,prev_done):
		next_done = []
		for idx in self.param.team_1_idxs:
			captured = np.any(self.dist_robots[idx,self.param.team_2_idxs] < self.param.tag_radius)
			reached_goal = self.dist_goal[idx] < self.param.tag_radius
			
			if idx in prev_done or captured: 
				next_done.append(idx)
			if reached_goal: 
				next_done = self.param.team_1_idxs
				break
		return next_done

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



class Node:
	# shared variables for all instances
	param = None
	
	def __init__(self,state,parent=None,action_to_node=None):
		self.state = state 
		self.parent = parent 
		self.action_to_node = action_to_node 

		self.number_of_visits = 0.
		self.value_1 = 0.
		self.value_2 = 0.
		self.children = []
		self.children_weights = []

		self.untried_actions = self.state.possible_actions().copy()
		random.shuffle(self.untried_actions)

	def __repr__(self):
		children_list = [(w, c.action_to_node) for w, c in zip(self.children_weights, self.children)]
		children_list.sort(key = lambda x: x[0], reverse=True)
		return "Node(state={}, n={}, v1={}, v2={}, children={})".format(self.state, self.number_of_visits, self.value_1, self.value_2, children_list)

	@classmethod
	def init_class(cls, param):
		cls.param = param

	def q(self, team_1_turn):
		value = self.value_1 if team_1_turn else self.value_2 
		return value

	def expand(self):
		while len(self.untried_actions) > 0:
			action = self.untried_actions.pop()
			next_state = self.state.forward(action)
			if next_state is not None:
				child_node = Node(next_state,parent=self,action_to_node=action)
				self.children.append(child_node)
				self.children_weights.append(None)
				return child_node

	def is_terminal(self):
		return self.state.is_terminal() or len(self.children) == 0

	def best_child(self, c_param=1.4):
		for k, c in enumerate(self.children): 
			self.children_weights[k] = (c.q(self.state.turn) / c.number_of_visits) + \
				c_param * np.sqrt((2 * np.log(self.number_of_visits) / c.number_of_visits))
		return self.children[np.argmax(self.children_weights)]


class Tree:

	def __init__(self,param):
		self.param = param
		self.root_node = None

		dynamicsA = RobotDynamics(param, 'a')
		dynamicsB = RobotDynamics(param, 'b')
		State.init_class(param, dynamicsA, dynamicsB)
		Node.init_class(param)

	def __repr__(self):
		return "Tree(root_node={})".format(self.root_node)

	def grow(self):	
		while self.root_node.number_of_visits < self.param.tree_size:
			current_node = self.tree_policy()
			reward_1,reward_2 = self.rollout(current_node)
			self.backpropagate(current_node,reward_1,reward_2)

	def set_root(self,root_state):
		if self.root_node is None:
			self.root_node = Node(root_state)
			return

		# check if we already have the state in the tree (as direct child of root)
		for node in self.root_node.children:
			if node.state == root_state:
				self.root_node = node
				self.root_node.parent = None
				# the garbage collector should clean up the now otherwise
				# not connected children
				print("Re-use search tree with {} nodes".format(self.root_node.number_of_visits))
				return

		# not found = re-create root
		self.root_node = Node(root_state)

	def tree_policy(self):
		current_node = self.root_node
		while not current_node.state.is_terminal():
			child = current_node.expand()
			if child is None:
				if len(current_node.children) > 0:
					current_node = current_node.best_child()
				else:
					return current_node
			else:
				return child
		return current_node

	def best_action(self):
		if self.root_node.is_terminal():
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
			untried_actions = state.possible_actions().copy()
			random.shuffle(untried_actions)
			if state.is_terminal() or len(untried_actions) == 0: 
				return reward_1,reward_2
			while len(untried_actions) > 0:
				action = untried_actions.pop()
				next_state = state.forward(action)
				if next_state is not None:
					state = next_state
					break
				if len(untried_actions) == 0:
					return reward_1,reward_2
			reward_1,reward_2 = state.eval_reward()
		return reward_1,reward_2