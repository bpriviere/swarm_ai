

class Dynamics:

	def __init__(self,param):
		self.param = param 

		# in subclass init, define: 
		# 	- self.state_dim_per_agent
		# 	- self.control_dim_per_agent
		# 	- self.A
		# 	- self.B

	def forward_per_node(self,x_i,u_i):
		print('error forward per node not overwritten')
		exit()