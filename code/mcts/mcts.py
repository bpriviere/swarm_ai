
import numpy as np 
from collections import defaultdict
import itertools
import random 

import sys
sys.path.append("../")
from utilities import dbgp

# https://int8.io/monte-carlo-tree-search-beginners-guide/ 
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py 
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/search.py 

def sample(actions):
	return actions[np.random.randint(len(actions))]

def forward(state,action):

	# defn
	A = np.array([
		[1,0,state.param.sim_dt,0],
		[0,1,0,state.param.sim_dt],
		[0,0,1,0],
		[0,0,0,1]
		])
	B = np.array([
		[0,0],
		[0,0],
		[state.param.sim_dt,0],
		[0,state.param.sim_dt]])

	# robots_state
	if state.team_1_turn: 
		# idxs = list(set(state.param.team_1_idxs)-set([not i for i in state.robots_done]))
		idxs = state.param.team_1_idxs
	else:
		idxs = state.param.team_2_idxs

	next_robots_state = np.copy(state.robots_state)
	for idx_i in idxs:
		next_robots_state[idx_i,:] = np.dot(A,state.robots_state[idx_i,:]) + np.dot(B,action[idx_i,:])
	
	# done flags
	next_robots_done = np.zeros(state.param.num_nodes_A) 
	for idx_i in state.param.team_1_idxs:
		if state.robots_done[idx_i] or \
				state.dist_goal[idx_i] < state.param.tag_radius or \
				np.any(state.dist_robots[idx_i,state.param.team_2_idxs] < state.param.tag_radius):
			next_robots_done[idx_i] = True 
	
	# team_1_turn
	next_team_1_turn = not state.team_1_turn 

	# make next state 
	next_state = State(state.param,next_robots_state,next_robots_done,next_team_1_turn)
	return next_state

class State: 

	def __init__(self,param,robots_state,robots_done,team_1_turn):
		self.param = param 
		self.robots_state = robots_state
		self.robots_done = robots_done 
		self.team_1_turn = team_1_turn
		self.dist_robots = self.make_dist_robots()
		self.dist_goal = self.make_dist_goal()

	def make_dist_robots(self):
		dist_robots = np.zeros((self.robots_state.shape[0],self.robots_state.shape[0]))
		for idx_i,state_i in enumerate(self.robots_state):
			for idx_j,state_j in enumerate(self.robots_state):
				dist_robots[idx_i,idx_j] = np.linalg.norm(state_i[0:2]-state_j[0:2])
		return dist_robots 

	def make_dist_goal(self):
		dist_goal = np.zeros(self.param.num_nodes_A)
		for idx_i in self.param.team_1_idxs:
			dist_goal[idx_i] = np.linalg.norm(self.robots_state[idx_i,0:2] - self.param.goal)
		return dist_goal

	def get_legal_actions(self):
		if self.team_1_turn:
			idxs = self.param.team_1_idxs
			u_max = self.param.acceleration_limit_a 
			v_max = self.param.speed_limit_a 
		else: 
			idxs = self.param.team_2_idxs
			u_max = self.param.acceleration_limit_b 
			v_max = self.param.speed_limit_b 

		u_xs = u_max*np.asarray([-1,0,1])
		u_ys = u_max*np.asarray([-1,0,1])

		master_lst = [] 
		for idx in idxs:
			master_lst.extend([u_xs,u_ys])
		master_lst = list(itertools.product(*master_lst))

		actions = []
		for elem in list(master_lst):
			action = np.zeros((self.robots_state.shape[0],2))
			for action_idx,robot_idx in enumerate(idxs): 
				action[robot_idx,:] = np.array(elem)[action_idx*2 + np.arange(2)] 
			# todo: velocity check 			
			# for robot_idx in idxs:
			actions.append(action)
		return actions

	def evaluate_reward(self):
		reward_1,reward_2 = 0, 0
		for idx_i in self.param.team_1_idxs: 
			if self.dist_goal[idx_i] < self.param.tag_radius:
				reward_1 += 1 
				reward_2 += 0
			elif np.any(self.dist_robots[idx_i,self.param.team_2_idxs] < self.param.tag_radius):
				reward_1 += 0 
				reward_2 += 1
			else:
				reward_1 += np.exp(-1*self.dist_goal[idx_i])
				reward_2 += np.exp(-1*np.sum(self.dist_robots[idx_i,self.param.team_2_idxs]))
		reward_1 /= len(self.param.team_1_idxs)
		reward_2 /= len(self.param.team_1_idxs)
		return reward_1,reward_2			

class Node: 
	
	def __init__(self,param,state,parent=None,action_to_node=None):
		self.param = param
		self.state = state 
		self.parent = parent 
		self.action_to_node = action_to_node 
		self.untried_actions = self.state.get_legal_actions()
		random.shuffle(self.untried_actions)
		self.number_of_visits = 0.
		self.value_1 = 0.
		self.value_2 = 0.
		self.children = [] 

	@property
	def q(self):
		if self.parent.state.team_1_turn:
			value = self.value_1
		else:
			value = self.value_2 
		return value

	@property
	def n(self):
		return self.number_of_visits

	def is_terminal_node(self):
		return np.all(self.state.robots_done)

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def expand(self):
		action = self.untried_actions.pop()
		next_state = forward(self.state,action)
		child_node = Node(self.param,next_state,parent=self,action_to_node=action)
		self.children.append(child_node)
		return child_node

	def best_child(self, c_param=None):
		if c_param is None:
			c_param = self.param.c_param 

		weights = []
		for c in self.children: 
			weight = (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
			weights.append(weight)
		return self.children[np.argmax(weights)]	


class Tree: 

	def __init__(self,param):
		self.param = param 
		self.num_nodes = 0 
		self.root_node = None

	def grow(self):	
		# while self.num_nodes < self.param.tree_size:
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

	def remove_branch(self,node,save_node):
		if not node is save_node: 
			for child in node.children: 
				self.remove_branch(child,save_node)
			self.remove_node(node)

	def remove_node(self,node):
		self.num_nodes = self.num_nodes - 1 
		del node 

	def tree_policy(self):
		current_node = self.root_node
		while not current_node.is_terminal_node():
			if not current_node.is_fully_expanded():
				child_node = current_node.expand()
				self.num_nodes += 1 
				return child_node
			else:
				current_node = current_node.best_child()
		return current_node

	def best_action(self):
		if self.root_node.is_terminal_node():
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
		value_1,value_2 = 0,0
		for i in range(node.param.rollout_horizon):

			actions = state.get_legal_actions()
			action = sample(actions)
			state = forward(state,action)
			reward_1,reward_2 = state.evaluate_reward()

			value_1 = reward_1
			value_2 = reward_2

			if np.all(state.robots_done):
				print('breaking at i/rollout_horizon = {}/{}'.format(i,node.param.rollout_horizon))
				break 

		return value_1,value_2
