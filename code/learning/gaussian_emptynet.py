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

class GaussianEmptyNet(nn.Module):

	def __init__(self,param,device):
		super(GaussianEmptyNet, self).__init__()

		self.device = torch.device(device)
		self.acceleration_limit = param.robot_types["standard_robot"]["acceleration_limit"]
		self.r_sense = param.robot_types["standard_robot"]["r_sense"]

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
			param.l_conditional_network_architecture,
			param.l_network_activation,
			device)

		self.policy = FeedForward(
			param.l_policy_network_architecture,
			param.l_network_activation,
			device)	

		self.value = FeedForward(
			param.l_value_network_architecture,
			param.l_network_activation,
			device)		

		self.to(self.device)


	def to(self, device):
		self.device = device
		self.model_team_a.to(device)
		self.model_team_b.to(device)
		self.psi.to(device)
		self.policy.to(device)
		self.value.to(device)
		return super().to(device)

	def __call__(self,o_a,o_b,goal):

		batch_size = o_a.shape[0]

		# project goal to sensing area
		goal_norm = goal.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(goal_norm/self.r_sense,min=1)
		goal = torch.mul(scale.unsqueeze(1), goal)

		# condition on game state 
		output_rho_team_a = self.model_team_a(o_a)
		output_rho_team_b = self.model_team_b(o_b)
		y = torch.cat((output_rho_team_a, output_rho_team_b, goal),1)
		y = self.psi(y) 

		# value uses game state condition
		value = (torch.tanh(self.value(y))+1) / 2 

		# predict mean and variance 
		action_dim = 2 
		dist = self.policy(y) 
		mu = dist[:,0:action_dim]
		logvar = dist[:,action_dim:]		
		sd = torch.sqrt(torch.exp(logvar))

		# reparameterization trick 
		eps = torch.randn(size=(batch_size,action_dim),device=self.device)
		policy = mu + sd * eps

		# scale policy 
		policy_norm = policy.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(policy_norm/self.acceleration_limit,min=1)
		policy = torch.mul(scale.unsqueeze(1), policy)

		return value, policy 
