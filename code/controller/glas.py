
import numpy as np 
import torch 
import sys

sys.path.append("../")
from measurements.relative_state import relative_state
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

	def policy(self,estimate):
		nodes = self.env.nodes
		actions = dict() 
		observations = relative_state(self.env.nodes, self.param.r_sense, self.param.goal, flatten=True)
		for node in nodes:
			
			o_a, o_b, goal = observations[node]
			o_a = torch.from_numpy(np.expand_dims(o_a,axis=0)).float() 
			o_b = torch.from_numpy(np.expand_dims(o_b,axis=0)).float()
			goal = torch.from_numpy(np.expand_dims(goal,axis=0)).float()

			if node.idx in self.param.team_1_idxs: 
				classification = self.model_A(o_a,o_b,goal).detach().numpy().T # 9 x 1 
				actions[node] = self.param.acceleration_limit_a/np.sqrt(2)*self.param.actions[np.argmax(classification)][np.newaxis].T # 2x1   
			elif node.idx in self.param.team_2_idxs: 
				classification = self.model_B(o_a,o_b,goal).detach().numpy().T # 9 x 1 
				actions[node] = self.param.acceleration_limit_b/np.sqrt(2)*self.param.actions[np.argmax(classification)][np.newaxis].T # 2x1  
		
			print('team {}, classification {}:'.format(node.team_A,classification))
		exit()

		return actions