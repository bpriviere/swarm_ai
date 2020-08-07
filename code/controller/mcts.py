import numpy as np 
import sys 

sys.path.append("../")
from mcts import mcts 
from controller.controller import Controller
	
class Controller(Controller):

	def __init__(self,param,env):
		super(Controller, self).__init__(param,env)
		self.tree = mcts.Tree(param)
		self.done = []

	def policy(self,estimate):

		state_mat = np.zeros((self.param.num_nodes,4))
		for node, estimate_i in estimate.items():
			state_mat[node.idx,:] = self.env.state_dict[node].flatten()
			# state_mat[node.idx,:] = estimate_i.state_mean.flatten()

		turn = True 
		done = self.done 

		actions_mat = np.zeros((self.param.num_nodes,2)) 
		
		for team in range(2): 
			tree_state = mcts.State(state_mat,done,turn)
			self.tree.set_root(tree_state)
			self.tree.grow() 
			next_tree_state, action = self.tree.best_action() 

			if action is None: 
				self.done = [] 
				break 

			state_mat = next_tree_state.state
			actions_mat += action
			turn = not turn 
			done = next_tree_state.done 

		actions = dict() 
		for node, estimate_i in estimate.items():
			actions[node] = actions_mat[node.idx,:,np.newaxis]
		return actions

	def state_vec_to_state_mat(self,state):
		return np.reshape(state,(param.num_nodes,4))