# standard package
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
import concurrent

# my package
from learning.deepset import DeepSet
from learning.feedforward import FeedForward

class EmptyNet(nn.Module):

	def __init__(self,param,device):
		super(EmptyNet, self).__init__()

		self.device = device
		self.param = param 

		self.model_team_a = DeepSet(
			param.il_phi_network_architecture,
			param.il_rho_network_architecture,
			param.il_network_activation,device)

		self.model_team_b = DeepSet(
			param.il_phi_network_architecture,
			param.il_rho_network_architecture,
			param.il_network_activation,device)		

		self.psi = FeedForward(
			param.il_psi_network_architecture,
			param.il_network_activation,device)

		self.param = param
		self.device = torch.device('cpu')

		# self.dim_neighbor = param.il_phi_network_architecture[0].in_features
		# self.dim_action = param.il_psi_network_architecture[-1].out_features
		# self.dim_state = param.il_psi_network_architecture[0].in_features - \
		# 				param.il_rho_network_architecture[-1].out_features - \
		# 				param.il_rho_obs_network_architecture[-1].out_features


	def to(self, device):
		self.device = device
		self.model_team_a.to(device)
		self.model_team_b.to(device)
		self.psi.to(device)
		return super().to(device)


	def __call__(self,o_a,o_b):

		output_rho_team_a = self.model_team_a(o_a)
		output_rho_team_b = self.model_team_b(o_b)

		x = torch.cat((output_rho_team_a, output_rho_team_b),1)
		x = self.psi(x)

		return x


	# def save_weights(self, filename):
	# 	torch.save({
	# 		'neighbors_phi_state_dict': self.model_neighbors.phi.state_dict(),
	# 		'neighbors_rho_state_dict': self.model_neighbors.rho.state_dict(),
	# 		'obstacles_phi_state_dict': self.model_obstacles.phi.state_dict(),
	# 		'obstacles_rho_state_dict': self.model_obstacles.rho.state_dict(),
	# 		'psi_state_dict': self.psi.state_dict(),
	# 		}, filename)


	# def load_weights(self, filename):
	# 	checkpoint = torch.load(filename)
	# 	self.model_neighbors.phi.load_state_dict(checkpoint['neighbors_phi_state_dict'])
	# 	self.model_neighbors.rho.load_state_dict(checkpoint['neighbors_rho_state_dict'])
	# 	self.model_obstacles.phi.load_state_dict(checkpoint['obstacles_phi_state_dict'])
	# 	self.model_obstacles.rho.load_state_dict(checkpoint['obstacles_rho_state_dict'])
	# 	self.psi.load_state_dict(checkpoint['psi_state_dict'])


	# def policy(self,x):

	# 	# inputs observation from all agents...
	# 	# outputs policy for all agents
	# 	grouping = dict()
	# 	for i,x_i in enumerate(x):
	# 		key = (int(x_i[0][0]), x_i.shape[1])
	# 		if key in grouping:
	# 			grouping[key].append(i)
	# 		else:
	# 			grouping[key] = [i]

	# 	A = np.empty((len(x),self.dim_action))
	# 	for key, idxs in grouping.items():
	# 		batch = torch.Tensor([x[idx][0] for idx in idxs])
	# 		a = self(batch)
	# 		a = a.detach().numpy()
	# 		for i, idx in enumerate(idxs):
	# 			A[idx,:] = a[i]

	# 	return A


	# def export_to_onnx(self, filename):
	# 	self.model_neighbors.export_to_onnx("{}_neighbors".format(filename))
	# 	self.model_obstacles.export_to_onnx("{}_obstacles".format(filename))
	# 	self.psi.export_to_onnx("{}_psi".format(filename))

	# def get_num_neighbors(self,x):
	# 	return int(x[0,0])

	# def get_num_obstacles(self,x):
	# 	nn = self.get_num_neighbors(x)
	# 	return int((x.shape[1] - 1 - self.dim_state - nn*self.dim_neighbor) / 2)  # number of obstacles 

	# def get_agent_idx_all(self,x):
	# 	nn = self.get_num_neighbors(x)
	# 	idx = np.arange(1+self.dim_state,1+self.dim_state+self.dim_neighbor*nn,dtype=int)
	# 	return idx

	# def get_obstacle_idx_all(self,x):
	# 	nn = self.get_num_neighbors(x)
	# 	idx = np.arange((1+self.dim_state)+self.dim_neighbor*nn, x.size()[1],dtype=int)
	# 	return idx

	# def get_goal_idx(self,x):
	# 	idx = np.arange(1,1+self.dim_state,dtype=int)
	# 	return idx 


