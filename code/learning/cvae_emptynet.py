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

class CVAE_EmptyNet(nn.Module):

	def __init__(self,param,device):
		super(CVAE_EmptyNet, self).__init__()

		self.device = device
		self.param = param 
		self.actions = torch.tensor(param.actions)

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

		self.decoder = FeedForward(
			param.il_decoder_network_architecture,
			param.il_network_activation,device)

		self.param = param
		self.device = torch.device('cpu')

		self.latent_dim = il_decoder_network_architecture


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

		sigma = x[]
		mu = x[]

		xbar = np.random.multivariate_normal(np.zeros(self.latent_dim),\
			np.eye(self.latent_dim,self.latent_dim))

		sample = torch.dot(sigma,xbar) + mu

		action = self.decoder(sample)
		
		return action

	def class_to_actions(self,x):
		# batch size is 0th dim
		actions = torch.zeros((x.shape[0],2),device=self.device) 

		for i in range(x.shape[0]):
			actions[i,:] = self.actions[x[i],:]
		return actions 
