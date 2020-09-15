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

class ContinuousEmptyNet(nn.Module):

	def __init__(self,param,device):
		super(ContinuousEmptyNet, self).__init__()

		self.device = device
		self.param = param 

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

		self.psi = FeedForward(
			param.l_psi_network_architecture,
			param.l_network_activation,device)

		self.encoder = FeedForward(
			param.l_encoder_network_architecture,
			param.l_network_activation,device)

		self.decoder = FeedForward(
			param.l_decoder_network_architecture,
			param.l_network_activation,device)

		self.param = param
		self.device = torch.device('cpu')


	def to(self, device):
		self.device = device
		self.model_team_a.to(device)
		self.model_team_b.to(device)
		self.psi.to(device)
		return super().to(device)

	def __call__(self,o_a,o_b,goal,training=False):

		output_rho_team_a = self.model_team_a(o_a)
		output_rho_team_b = self.model_team_b(o_b)

		x = torch.cat((output_rho_team_a, output_rho_team_b, goal),1)

		value = (torch.tanh(self.psi(x))+1) / 2 

		# 'reparameterization trick' : https://towardsdatascience.com/reparameterization-trick-126062cfd3c3
		x = self.encoder(x)
		batch_size = x.shape[0]
		z_dim = int(x.shape[1]/2)
		z_mu = x[:,0:z_dim]
		z_logvar = x[:,z_dim:]
		eps = torch.randn(size=(batch_size,z_dim))
		z = z_mu + torch.exp(z_logvar / 2) * eps
		policy = self.decoder(z)

		if training:
			return value, policy, z_mu, z_logvar
		else:
			return value, policy