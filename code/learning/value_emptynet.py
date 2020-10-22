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

class ValueEmptyNet(nn.Module):

	def __init__(self,param,device):
		super(ValueEmptyNet, self).__init__()

		self.device = torch.device(device)

		self.model_team_a = DeepSet(
			param.l_phi_network_architecture,
			param.l_rho_network_architecture,
			param.l_network_activation,
			device)

		self.model_team_b = DeepSet(
			param.l_phi_network_architecture,
			param.l_rho_network_architecture,
			param.l_network_activation,
			device)

		self.value = FeedForward(
			param.l_xi_network_architecture,
			param.l_network_activation,
			device)

		self.to(self.device)


	def to(self, device):
		self.device = device
		self.model_team_a.to(device)
		self.model_team_b.to(device)
		self.value.to(device)
		return super().to(device)

	def __call__(self,v_a,v_b,num_a,num_b,num_rg,training=False):

		output_rho_team_a = self.model_team_a(v_a)
		output_rho_team_b = self.model_team_b(v_b)

		y = torch.cat((output_rho_team_a,output_rho_team_b,num_a,num_b,num_rg),1)
		
		# predict mean and variance 
		dist = self.value(y)
		mu = torch.unsqueeze(dist[:,0],1)
		logvar = torch.unsqueeze(dist[:,1],1)

		if training: 
			return None, mu, logvar 
		else: 

			batch_size = v_a.shape[0]

			# reparameterization trick 
			sd = torch.sqrt(torch.exp(logvar))
			eps = torch.randn(size=(batch_size,1),device=self.device)
			value = mu + sd * eps

			# scale 
			value = (torch.tanh(value)+1)/2

			return value
