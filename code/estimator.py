import numpy as np 

class Estimator: 

	def __init__(self,param,env):
		self.env = env
		self.param = param 


	def estimate(self,observations):

		estimates = dict()

		for node in self.env.nodes: 
			estimate = np.zeros((self.param.state_dim_per_agent,1))
			estimates[node] = estimate 

		return estimates 
