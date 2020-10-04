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

		self.device = torch.device(device)

		# self.acceleration_limit = param.standard_robot["acceleration_limit"]
		self.acceleration_limit = param.robot_types["standard_robot"]["acceleration_limit"]
		self.r_sense = param.robot_types["standard_robot"]["r_sense"]

		self.z_dim = param.l_z_dim

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

		self.encoder = FeedForward(
			param.l_encoder_network_architecture,
			param.l_network_activation,
			device)

		self.decoder = FeedForward(
			param.l_decoder_network_architecture,
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
		self.encoder.to(device)
		self.decoder.to(device)
		self.value.to(device)
		return super().to(device)

	def __call__(self,o_a,o_b,goal,x=None):

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

		if x is None: 
			# simple sample 
			mu = 0.0 
			sd = 1.0 

		else:
			# encode action 
			dist = self.encoder(torch.cat((x,y),1))
			mu = dist[:,0:self.z_dim]
			# sd = dist[:,self.z_dim:]
			logvar = dist[:,self.z_dim:]
			sd = torch.pow(torch.exp(logvar),1/2)

		eps = torch.randn(size=(batch_size,self.z_dim),device=self.device)
		z = mu + sd * eps

		# decode 
		policy = self.decoder(torch.cat((z,y),1))

		# scale policy 
		policy_norm = policy.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(policy_norm/self.acceleration_limit,min=1)
		policy = torch.mul(scale.unsqueeze(1), policy)

		# value uses game state condition
		value = (torch.tanh(self.value(y))+1) / 2 

		if x is None:
			return value, policy
		else: 
			return value, policy, mu, logvar


	# def torch_scale(self,action,max_action):
	# 	action_norm = action.norm(p=2,dim=1)
	# 	index = action_norm > 0
	# 	scale = torch.ones(action.shape[0],device=self.device)
	# 	scale[index] = 1.0 / torch.clamp(action_norm[index]/max_action,min=1)
	# 	action = torch.mul(scale.unsqueeze(1), action)
	# 	return action