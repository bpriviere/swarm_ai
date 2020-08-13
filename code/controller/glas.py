
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

	def policy(self,estimate):

		with torch.no_grad():

			nodes = self.env.nodes
			actions = dict() 

			observations = relative_state_per_node(self.env.nodes, self.env.state_vec_to_mat(self.env.state_vec), self.param)
			for node in nodes:
				
				o_a, o_b, goal = observations[node]
				if o_a.shape[0] == 0:
					o_a = np.expand_dims(o_a,axis=0)
				if o_b.shape[0] == 0:
					o_b = np.expand_dims(o_b,axis=0)

				o_a = torch.from_numpy(o_a).float() 
				o_b = torch.from_numpy(o_b).float()
				goal = torch.from_numpy(np.expand_dims(goal,axis=0)).float()

				if node.idx in self.param.team_1_idxs: 
					classification = self.model_A(o_a,o_b,goal).detach().numpy().T # 9 x 1 
				elif node.idx in self.param.team_2_idxs: 
					classification = self.model_B(o_a,o_b,goal).detach().numpy().T # 9 x 1 

				# idx = np.argmax(classification)
				idx = np.random.choice(range(len(self.param.actions)),p=classification.flatten())
				actions[node] = node.acceleration_limit/np.sqrt(2)*self.param.actions[idx][np.newaxis].T # 2x1   
			return actions