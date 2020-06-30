import numpy as np 
import importlib
import signal
import json 
import pprint

def dbgp(name,value):
	# debug_print
	if type(value) is dict:
		print('{}'.format(name))
		for key_i,value_i in value.items():
			print('{}:{}'.format(str(key_i),value_i))
	else:
		print('{}:{}'.format(name,value))

def load_module(fn):
	module_dir, module_name = fn.split("/")
	module_name, _ = module_name.split(".")
	module = importlib.import_module("{}.{}".format(module_dir, module_name))
	return module	

class Network: 
	def __init__(self,nodes,edges):
		self.nodes = nodes
		self.edges = edges
		self.adjacency = self.get_adjacency_matrix()
		self.degree = self.get_degree_matrix()
		self.laplacian = self.get_laplacian_matrix()
		self.num_connected_components = self.get_num_connected_components()

	def get_adjacency_matrix(self):
		A = np.zeros((len(self.nodes),len(self.nodes)))
		for node_i,node_j in self.edges:
			A[node_i.idx,node_j.idx] = 1.0 
		return A 

	def get_degree_matrix(self):
		D = np.zeros((len(self.nodes),len(self.nodes)))
		for node_i in self.nodes:
			D[node_i.idx,node_i.idx] = sum(self.adjacency[node_i.idx,:])
		return D 

	def get_laplacian_matrix(self):
		L = self.degree - self.adjacency 
		return L 

	def get_num_connected_components(self):
		l,v = np.linalg.eig(self.laplacian)
		l = np.round(l,5)
		num_connected_components = np.count_nonzero(l==0)
		return num_connected_components	