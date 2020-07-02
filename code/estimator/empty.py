
import numpy as np 
from estimator.estimator import Estimator, Estimate

class Estimate(Estimate):

	def __init__(self,state_mean,state_covariance):
		super(Estimate, self).__init__(state_mean, state_covariance):

	def update_node(self,node):
		node.state_mean = self.state_mean 
		node.state_covariance = self.state_covariance
		return node 


class Estimator(Estimator): 

	def __init__(self,param,env):
		super(Estimator, self).__init__(param,env)

	def estimate(self,observations,actions):

		estimates = dict()

		for node in self.env.nodes: 
			estimate = np.zeros((self.param.state_dim_per_agent,1))
			estimates[node] = estimate 

		return estimates 