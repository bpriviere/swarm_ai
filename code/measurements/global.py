
import numpy as np 

from measurements.measurements import Measurements
	
class Measurements(Measurements):

	def __init__(self,param):
		super(Measurements, self).__init__(param)
		

	def measure_per_node(self,x):

		state_dim = x.shape[0]
		measurement_dim = state_dim

		C_i = np.eye((state_dim))

		# noise 
		R_i = self.param.measurement_noise_covariance * np.eye(measurement_dim)
		v_i = np.dot(R_i,np.random.normal(size=((measurement_dim,1))))
		
		# solve 
		y_i = np.dot(C_i,x) + v_i

		return y_i, C_i, R_i