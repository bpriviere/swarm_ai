
import numpy as np 
from attacker.attacker import Attacker
	
class Attacker(Attacker):

	def __init__(self,param,env):
		super(Attacker, self).__init__(param,env)

	def attack(self,observations):
		return observations

