class Attacker: 

	def __init__(self,param,env):
		self.param = param 
		self.env = env 


	def attack(self,observations):
		return observations