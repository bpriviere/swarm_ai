
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
import numpy as np
from learning.feedforward import FeedForward

class DeepSet(nn.Module):


	def __init__(self,phi_layers,rho_layers,activation,device):
		super(DeepSet, self).__init__()
		
		self.phi = FeedForward(phi_layers,activation,device)
		self.rho = FeedForward(rho_layers,activation,device)
		self.device = torch.device(device)


	def to(self, device):
		self.device = device
		self.phi.to(device)
		self.rho.to(device)
		return super().to(device)


	def export_to_onnx(self, filename):
		self.phi.export_to_onnx("{}_phi".format(filename))
		self.rho.export_to_onnx("{}_rho".format(filename))


	def forward(self,x):
		# x is list of relative state measurements to members on team a (or b)

		X = torch.zeros((len(x),self.rho.in_dim), device=self.device)
		num_elements = int(x.size()[1] / self.phi.in_dim)
		for i in range(num_elements):
			X += self.phi(x[:,i*self.phi.in_dim:(i+1)*self.phi.in_dim])
		return self.rho(X)