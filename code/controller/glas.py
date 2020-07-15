
import numpy as np 
import torch 
import sys

sys.path.append("../")
from measurements.relative_state import relative_state
from controller.controller import Controller
from controller.joint_mpc import Controller as MPC
from learning.emptynet import EmptyNet
from glas.gparam import Gparam 	

class Controller(Controller):

	def __init__(self,param,env):
		super(Controller, self).__init__(param,env)

		gparam = Gparam() 
		device = "cpu"
		self.model = EmptyNet(gparam, device)
		self.model.load_state_dict(torch.load(self.param.glas_model))
		self.MPC = MPC(param,env)


	def policy(self,estimate):

		# team A policy is mpc
		states_A, actions_A = self.MPC.policy_A(self.env.nodes,estimate)

		# team B policy is glas 
		actions_B = self.policy_B(self.env.nodes,estimate)

		# recombine 
		actions = dict()
		for node in self.env.nodes:
			if node.team_A:
				actions[node] = np.expand_dims(actions_A[node][:,0],axis=1)
			elif node.team_B:
				actions[node] = actions_B[node]

		return actions 


	def policy_B(self,nodes,estimate):

		nodes_B = []
		for node in nodes: 
			if node.team_B: 
				nodes_B.append(node)

		actions = dict() 
		observations = relative_state(self.env.nodes, self.param.r_sense,flatten=True)
		for node in nodes_B:
			o_a, o_b = observations[node]
			o_a = torch.from_numpy(np.expand_dims(o_a,axis=0)).float() 
			o_b = torch.from_numpy(np.expand_dims(o_b,axis=0)).float()
			actions[node] = self.model(o_a,o_b).detach().numpy().T # 2 x 1 
		return actions