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

class PolicyEmptyNet(nn.Module):

	def __init__(self,param,device):
		super(PolicyEmptyNet, self).__init__()

		self.action_dim = param.dynamics["control_dim"]
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

		self.policy = FeedForward(
			param.l_policy_network_architecture,
			param.l_network_activation,
			device)	

		if param.dynamics["name"] == "double_integrator":
			self.scale = self.scale_di
		elif param.dynamics["name"] == "single_integrator":
			self.scale = self.scale_si
		elif param.dynamics["name"] == "dubins_2d":
			self.scale = self.scale_dubins_2d
		elif param.dynamics["name"] == "dubins_3d":
			self.scale = self.scale_dubins_3d

		self.to(self.device)


	def to(self, device):
		self.device = device
		self.model_team_a.to(device)
		self.model_team_b.to(device)
		self.policy.to(device)
		return super().to(device)

	def __call__(self,o_a,o_b,goal,training=False):

		# project goal to sensing area
		goal_norm = goal.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(goal_norm/self.r_sense,min=1)
		goal = torch.mul(scale.unsqueeze(1), goal)

		# condition on game state 
		output_rho_team_a = self.model_team_a(o_a)
		output_rho_team_b = self.model_team_b(o_b)
		y = torch.cat((output_rho_team_a, output_rho_team_b, goal),1)
		
		# predict mean and variance 
		action_dim = self.action_dim
		dist = self.policy(y)
		mu = dist[:,0:action_dim]
		logvar = dist[:,action_dim:]

		if training:
			return None, mu, logvar
		else:
			batch_size = o_a.shape[0]

			sd = torch.sqrt(torch.exp(logvar))

			# reparameterization trick 
			eps = torch.randn(size=(batch_size,action_dim),device=self.device)
			policy = mu + sd * eps

			# scale policy 
			policy = self.scale(policy)

			return policy 

	def scale_di(self,policy):
		policy_norm = policy.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(policy_norm/self.acceleration_limit,min=1)
		policy = torch.mul(scale.unsqueeze(1), policy)
		return policy 

	def scale_si(self,policy):
		policy_norm = policy.norm(p=2,dim=1)
		scale = 1.0 / torch.clamp(policy_norm/self.acceleration_limit,min=1)
		policy = torch.mul(scale.unsqueeze(1), policy)
		return policy 

	def scale_dubins_2d(self,policy):
		# policy in bs x 2 : 
		exit('scale in policy_emptynet not defined...')
		l = torch.tensor([[0.2, 0.3, 0.]])
		u = torch.tensor([[0.8, 1., 0.65]])
		policy = torch.max(torch.min(policy, u), l)
		return policy 

	def scale_dubins_3d(self,policy):
		# policy in bs x 3 : phidot,psidot,vdot
		angular_acc_lim = 2.0*np.pi/5.0
		l = torch.tensor([[-angular_acc_lim, -angular_acc_lim, -self.acceleration_limit]])
		u = torch.tensor([[angular_acc_lim, angular_acc_lim, self.acceleration_limit]])
		policy = torch.max(torch.min(policy, u), l)
		return policy 