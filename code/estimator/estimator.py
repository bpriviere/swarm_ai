
class Estimator:

	def __init__(self,param,env):
		self.env = env
		self.param = param
		
	def estimate(self,observation,action):
		exit('estimate fnc not overwritten')

	def initial_estimate(self):

		estimate = dict()
		for node in self.env.nodes: 
			estimate[node] = Estimate(node.state_mean,node.state_covariance)
		return estimate 


class Estimate:

	def __init__(self, state_mean, state_covariance):
		self.state_mean = state_mean
		self.state_covariance = state_covariance
		
	def update_node(self,node):
		exit('update_node fnc not overwritten')

	


