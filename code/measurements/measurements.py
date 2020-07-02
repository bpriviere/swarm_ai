
class Measurements:

	def __init__(self,param):
		self.param = param 

		# in subclass init, define: 
		# 	- self.state_dim_per_agent
		# 	- self.control_dim_per_agent
		# 	- self.A
		# 	- self.B

	def measure_per_node(self,x):
		print('error measure_per_node not overwritten')
		exit()