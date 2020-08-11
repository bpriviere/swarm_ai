
import numpy as np 
import torch 
import sys

sys.path.append("../")
from measurements.relative_state import relative_state,relative_state_per_node
from controller.controller import Controller
from controller.joint_mpc import Controller as MPC
from learning.emptynet import EmptyNet
from learning.discrete_emptynet import DiscreteEmptyNet
from glas.gparam import Gparam 	

class Controller(Controller):

	def __init__(self,param,env):
		super(Controller, self).__init__(param,env)

		gparam = Gparam() 
		device = "cpu"
		self.model_A = DiscreteEmptyNet(gparam, device)
		self.model_A.load_state_dict(torch.load(self.param.glas_model_A))
		self.model_B = DiscreteEmptyNet(gparam, device)
		self.model_B.load_state_dict(torch.load(self.param.glas_model_B))

		# for p in self.model_A.psi.layers[0].parameters():
		# 	print(p)

		# for p in self.model_B.psi.layers[0].parameters():
		# 	print(p)	

		# exit()

	def policy(self,estimate):

		nodes = self.env.nodes
		actions = dict() 

		observations = relative_state_per_node(self.env.nodes, self.env.state_vec_to_mat(self.env.state_vec), self.param, flatten=True)

		for node in nodes:
			
			o_a, o_b, goal = observations[node]
			o_a = torch.from_numpy(np.expand_dims(o_a,axis=0)).float() 
			o_b = torch.from_numpy(np.expand_dims(o_b,axis=0)).float()
			goal = torch.from_numpy(np.expand_dims(goal,axis=0)).float()

			# print('o_a',o_a)
			# print('o_b',o_b)
			# print('goal',goal)

			if node.idx in self.param.team_1_idxs: 
				classification = self.model_A(o_a,o_b,goal).detach().numpy().T # 9 x 1 
			elif node.idx in self.param.team_2_idxs: 
				classification = self.model_B(o_a,o_b,goal).detach().numpy().T # 9 x 1 
			
			# idx = np.argmax(classification)
			idx = np.random.choice(range(len(self.param.actions)),p=classification.flatten())

			actions[node] = node.acceleration_limit/np.sqrt(2)*self.param.actions[idx][np.newaxis].T # 2x1  
			
			# print('classification',classification)
			# print('actions[node]',actions[node])
		# exit()
		
		return actions