
import numpy as np 
from collections import defaultdict
import itertools

import sys
sys.path.append("../")
from utilities import dbgp

# https://int8.io/monte-carlo-tree-search-beginners-guide/ 
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/nodes.py 
# https://github.com/int8/monte-carlo-tree-search/blob/master/mctspy/tree/search.py 

# team 1 trying to maximize value 
# team 2 trying to minimize value 

# parameters
r_capture = 0.05 
goal = np.array([0.45,0.05]) 
u_max_1 = 0.1 
u_max_2 = 0.2 
v_max_1 = 0.1 
v_max_2 = 0.2 
dt 		= 0.25 
num_simulations = 100
rollout_horizon = 10 
# team_1_idxs = [0] 
# team_2_idxs = [1] 
# team_1_idxs = [0,1] 
# team_2_idxs = [2] 
team_1_idxs = [0] 
team_2_idxs = [1,2] 
c_param = 1.4 

class Node: 
	
	def __init__(self,state,team_1_turn,parent=None,action_to_node=None):
		self.state = state 
		self.team_1_turn = team_1_turn 
		self.parent = parent 
		self.action_to_node = action_to_node 
		self.done, self.reward_1, self.reward_2 = evaluate_game(self.state) 
		self.untried_actions = get_legal_actions(self.state,self.team_1_turn) 
		self.number_of_visits = 0.
		self.value_1 = 0.
		self.value_2 = 0.
		self.children = [] 

	@property
	def q(self):
		if self.parent.team_1_turn:
			value = self.value_1
		else:
			value = self.value_2 
		return value

	@property
	def n(self):
		return self.number_of_visits

	def is_terminal_node(self):
		return self.done

	def is_fully_expanded(self):
		return len(self.untried_actions) == 0

	def expand(self):
		action = self.untried_actions.pop()
		next_state = forward(self.state,action,self.team_1_turn)
		next_turn = not self.team_1_turn 
		child_node = Node(next_state,next_turn,parent=self,action_to_node=action)
		self.children.append(child_node)
		return child_node

	def best_child(self, c_param=c_param):
		weights = []
		for c in self.children: 
			weight = (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
			weights.append(weight)
			# dbgp('weight',weight)
			# dbgp('c.action_to_node',c.action_to_node)
		return self.children[np.argmax(weights)]


def sample(actions):
	return actions[np.random.randint(len(actions))]


def evaluate_game(state):

	# print('state',state)

	dist_robots = np.zeros((state.shape[0],state.shape[0]))
	for idx_i,state_i in enumerate(state):
		for idx_j,state_j in enumerate(state):
			dist_robots[idx_i,idx_j] = np.linalg.norm(state_i[0:2]-state_j[0:2])
	
	# print('dist_robots',dist_robots)

	reward_1,reward_2 = 0, 0
	done = np.zeros(len(team_1_idxs),dtype=bool)
	for idx_i in team_1_idxs: 
		dist_goal = np.linalg.norm(state[idx_i,0:2] - goal)
		if dist_goal < r_capture:
			done[idx_i] = True
			reward_1 += 1 
			reward_2 += 0

		elif np.any(dist_robots[idx_i,team_2_idxs] < r_capture):
			done[idx_i] = True
			reward_1 += 0 
			reward_2 += 1

		else:
			# composite_dist = dist_goal + 10/np.min(dist_robots[idx_i,team_2_idxs])
			# reward_1 += np.exp(-1*composite_dist)
			# reward_2 += -reward_1

			reward_1 += np.exp(-1*dist_goal)
			# reward_2 += np.exp(-1*np.min(dist_robots[idx_i,team_2_idxs]))
			reward_2 += np.exp(-1*np.sum(dist_robots[idx_i,team_2_idxs]))

	reward_1 /= len(team_1_idxs)
	reward_2 /= len(team_1_idxs)

	# print('reward_1',reward_1)
	# print('reward_2',reward_2)
		
	done = np.any(done)
	return done,reward_1,reward_2


def forward(state,action,team_1_turn):
	if team_1_turn: 
		idxs = team_1_idxs
	else:
		idxs = team_2_idxs

	next_state = np.copy(state)
	for idx_i in idxs:
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


def get_legal_actions(state,team_1_turn):
	if team_1_turn:
		idxs = team_1_idxs
		u_max = u_max_1 
		v_max = v_max_1 
	else: 
		idxs = team_2_idxs
		u_max = u_max_2 
		v_max = v_max_2 

	u_xs = u_max*np.asarray([-1,0,1])
	u_ys = u_max*np.asarray([-1,0,1])

	master_lst = [] 
	for idx in idxs:
		master_lst.extend([u_xs,u_ys])
	master_lst = list(itertools.product(*master_lst))

	actions = []
	for elem in list(master_lst):
		action = np.zeros((state.shape[0],2))
		for action_idx,robot_idx in enumerate(idxs): 
			action[robot_idx,:] = np.array(elem)[action_idx*2 + np.arange(2)] 
		actions.append(action)
	return actions


def backpropagate(node,reward_1,reward_2):
	node.number_of_visits += 1.
	node.value_1 += reward_1
	node.value_2 += reward_2
	if node.parent:
		backpropagate(node.parent,reward_1,reward_2)


def rollout(node):
	value_1,value_2 = 0,0
	state = node.state
	team_1_turn = node.team_1_turn
	for i in range(rollout_horizon):
		actions = get_legal_actions(state,team_1_turn)
		action = sample(actions)
		state = forward(state,action,team_1_turn)
		done,reward_1,reward_2 = evaluate_game(state)

		value_1 = reward_1
		value_2 = reward_2
		team_1_turn = not team_1_turn 
		
		if done:
			print('breaking at i/rollout_horizon = {}/{}'.format(i,rollout_horizon))
			break 

	return value_1,value_2


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
		return None,None

	for _ in range(num_simulations):
		current_node = tree_policy(root_node)
		reward_1,reward_2 = rollout(current_node)
		backpropagate(current_node,reward_1,reward_2)

	# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	# print('text')
	best_child = root_node.best_child(c_param=0.0)
	# exit()

	return best_child.state, best_child.action_to_node

