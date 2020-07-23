
import numpy as np 
from collections import defaultdict

import sys
sys.path.append("../")
from utilities import dbgp

# https://int8.io/monte-carlo-tree-search-beginners-guide/
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py 
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/search.py

# parameters
r_capture = 0.05 
goal = np.array([0.6,0]) 
u_max_1 = 0.1 
u_max_2 = 0.2 
v_max_1 = 0.1 
v_max_2 = 0.2 
dt 		= 0.25 
num_simulations = 500 
rollout_horizon = 10 
team_1_idxs = [0] 
team_2_idxs = [1] 
c_param = 1.4 

class Node: 
	
	def __init__(self,state,team_1_turn,parent=None,action_to_node=None):
		self.state = state 
		self.team_1_turn = team_1_turn 
		self.parent = parent 
		self.action_to_node = action_to_node 
		self.done, self.score = evaluate_game(self.state)
		self.untried_actions = get_legal_actions(self)
		self.number_of_visits = 0.
		self.value = 0.0 
		self.children = [] 

	@property
	def q(self):
		if self.parent.team_1_turn:
			return -1*self.value
		else:
			return self.value 

	@property
	def n(self):
		return self.number_of_visits

	def is_terminal_node(self):
		return self.done

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def expand(self):
		action = self.untried_actions.pop()
		next_state = forward(self.state,action)
		next_turn = self.team_1_turn == False
		child_node = Node(next_state,next_turn,parent=self,action_to_node=action)
		self.children.append(child_node)
		return child_node

	def best_child(self, c_param=c_param):
		weights = []
		for c in self.children: 
			weight = (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
			weights.append(weight)
		return self.children[np.argmax(weights)]


def sample(actions):
	return actions[np.random.randint(len(actions))]


def evaluate_game(state):

	dist_robots = np.zeros((state.shape[0],state.shape[0]))
	for idx_i,state_i in enumerate(state):
		for idx_j,state_j in enumerate(state):
			dist_robots[idx_i,idx_j] = np.linalg.norm(state_i[0:2]-state_j[0:2])
	
	dist_goal = np.zeros(len(team_1_idxs))
	for idx_i in team_1_idxs:
		dist_goal[idx_i] = np.linalg.norm(state[idx_i,0:2] - goal)

	score = 0 
	done = np.zeros(len(team_1_idxs),dtype=bool)
	for idx_i in team_1_idxs: 
		if dist_goal[idx_i] < r_capture:
			done[idx_i] = True
		elif np.any(dist_robots[idx_i,team_2_idxs] < r_capture):
			done[idx_i] = True
		score += dist_goal[idx_i] 

	done = np.all(done)
	return done, score


def forward(state,action):
	next_state = np.zeros((state.shape))
	for idx_i in range(state.shape[0]):
		next_state[idx_i,:] = forward_per_robot(state[idx_i,:],action[idx_i,:])
	return next_state


def forward_per_robot(state_i,action_i):
	# double integrator 
	A = np.array([
		[1,0,dt,0],
		[0,1,0,dt],
		[0,0,1,0],
		[0,0,0,1]
		])
	B = np.array([
		[0,0],
		[0,0],
		[dt,0],
		[0,dt]])
	next_state_i = np.dot(A,state_i) + np.dot(B,action_i)
	return next_state_i


def get_legal_actions(node):
	if node.team_1_turn:
		idx = 0 
		u_max = u_max_1 
		v_max = v_max_1 
	else: 
		idx = 1 
		u_max = u_max_2 
		v_max = v_max_2 

	actions = []
	u_xs = u_max*np.asarray([-1,0,1])
	u_ys = u_max*np.asarray([-1,0,1])
	for u_x in u_xs:
		for u_y in u_ys:
			action = np.zeros((node.state.shape[0],2))
			action[idx,:] = np.array((u_x,u_y))
			# velocity check 
			if np.linalg.norm(forward_per_robot(node.state[idx,:],action[idx,:])[2:]) < v_max:
				actions.append(action)

	return actions


def backpropagate(node,reward):
	node.number_of_visits += 1.
	node.value += reward
	if node.parent:
		backpropagate(node.parent,reward)


def rollout(node):
	for i in range(rollout_horizon):
		actions = get_legal_actions(node)
		action = sample(actions)
		state = forward(node.state,action)
		done,score = evaluate_game(node.state)
		if done:
			print('breaking at i/rollout_horizon = {}/{}'.format(i,rollout_horizon))
			break 
	return score


def tree_policy(root_node):
	current_node = root_node
	while not current_node.is_terminal_node():
		if not current_node.is_fully_expanded():
			return current_node.expand()
		else:
			current_node = current_node.best_child()
	return current_node


def best_action(root_state,team_1_turn):
	root_node = Node(root_state,team_1_turn)
	
	if root_node.is_terminal_node():
		return root_node.state,np.zeros((root_state.shape[0],2))

	for _ in range(num_simulations):
		current_node = tree_policy(root_node)
		reward = rollout(current_node)
		backpropagate(current_node,reward)

	best_child = root_node.best_child(c_param=0.0)

	return best_child.state, best_child.action_to_node

