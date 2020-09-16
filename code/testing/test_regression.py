
# standard
import time 
import torch 
import sys 
import numpy as np 
import matplotlib.pyplot as plt 

# custom
sys.path.append('../')
from cpp.buildRelease import mctscpp

# state to game 

def state_to_cpp_game_state(state,team_1_idxs,turn):

	attackers = [] 
	defenders = [] 
	for i_robot, robot_state in enumerate(state): 
		rs = mctscpp.RobotState(robot_state)
		if i_robot in team_1_idxs: 
			attackers.append(rs)
		else:
			defenders.append(rs)

	if turn == "a":
		turn = mctscpp.GameState.Turn.Attackers
	elif turn == "b": 
		turn = mctscpp.GameState.Turn.Defenders
	else: 
		exit('training team not recognized') 

	game_state = mctscpp.GameState(turn,attackers,defenders)

	game_state.depth = 0

	return game_state 


# robot team stuff 

def create_cpp_robot_type(robot_types, robot_type):
	p_min = [robot_types[robot_type]["env_xlim"][0], robot_types[robot_type]["env_ylim"][0]]
	p_max = [robot_types[robot_type]["env_xlim"][1], robot_types[robot_type]["env_ylim"][1]]
	velocity_limit = robot_types[robot_type]["speed_limit"]
	acceleration_limit = robot_types[robot_type]["acceleration_limit"]
	tag_radius = robot_types[robot_type]["tag_radius"]
	r_sense = robot_types[robot_type]["r_sense"]
	radius = robot_types[robot_type]["radius"]
	rt = mctscpp.RobotType(p_min,p_max,velocity_limit,acceleration_limit,tag_radius,r_sense,radius)
	return rt

def robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,team):
	types = [] 
	for robot_type, num in robot_team_composition[team].items():
		rt = create_cpp_robot_type(robot_types, robot_type)
		for _ in range(num):
			types.append(rt)
	return types


# glas stuff 

def loadGLAS(glas, file):
	state_dict = torch.load(file)

	loadFeedForwardNNWeights(glas.deepSetA.phi, state_dict, "model_team_a.phi")
	loadFeedForwardNNWeights(glas.deepSetA.rho, state_dict, "model_team_a.rho")
	loadFeedForwardNNWeights(glas.deepSetB.phi, state_dict, "model_team_b.phi")
	loadFeedForwardNNWeights(glas.deepSetB.rho, state_dict, "model_team_b.rho")
	loadFeedForwardNNWeights(glas.psi, state_dict, "psi")
	loadFeedForwardNNWeights(glas.encoder, state_dict, "encoder")
	loadFeedForwardNNWeights(glas.decoder, state_dict, "decoder")

	return glas

def loadFeedForwardNNWeights(ff, state_dict, name):
	l = 0
	while True:
		key1 = "{}.layers.{}.weight".format(name, l)
		key2 = "{}.layers.{}.bias".format(name, l)
		if key1 in state_dict and key2 in state_dict:
			ff.addLayer(state_dict[key1].numpy(), state_dict[key2].numpy())
		else:
			if l == 0:
				print("WARNING: No weights found for {}".format(name))
			break
		l += 1

if __name__ == '__main__':
	
	tree_sizes = [100,1000,10000]
	betas = [0.0, 0.25, 0.5]
	ecc = 1.4 
	num_trials = 5 
	dt = 0.1
	max_depth = 1000
	path_glas_model_a = 'test_a.pt'
	path_glas_model_b = 'test_b.pt'
	goal = np.array([0.375,0.375,0,0])

	# use a fixed seed
	seed = 1
	mctscpp.seed(seed)

	# robot team information 
	team_1_idxs = [0]
	robot_types = {
		'standard_robot' : {
			'speed_limit': 0.125,
			'acceleration_limit':0.125,
			'tag_radius': 0.025,
			'dynamics':'double_integrator',
			'r_sense': 1.0,
			'radius': 0.025,
			'env_xlim': [0,0.5],
			'env_ylim': [0,0.5],
		}
	}
	robot_team_composition = {
		'a': {'standard_robot':1},
		'b': {'standard_robot':1}
		}

	# make game 
	attackerTypes = robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,"a") 
	defenderTypes = robot_composition_to_cpp_robot_types(robot_team_composition,robot_types,"b") 
	game = mctscpp.Game(attackerTypes, defenderTypes, dt, goal, max_depth)
	loadGLAS(game.glasA, path_glas_model_a)
	loadGLAS(game.glasB, path_glas_model_b)	

	# intial state information 
	state = [
		[0.2,0.2,0,0],
		[0.4,0.2,0,0],
	]	

	# run 
	results = np.zeros((len(tree_sizes),len(betas),num_trials))
	count = 0 
	total = len(tree_sizes)*len(betas)*num_trials
	for i_trial in range(num_trials):
		for i_beta,beta in enumerate(betas):
			for i_tree_size,tree_size in enumerate(tree_sizes):

				print('{}/{}'.format(count,total))

				game_state = state_to_cpp_game_state(state,team_1_idxs,"a")

				start = time.time()
				mctscpp.search(game, game_state, tree_size, beta, ecc)
				elapsed = time.time() - start 
				results[i_tree_size,i_beta,i_trial] = elapsed 

				count += 1 

	fig,ax = plt.subplots() 
	im = ax.imshow(np.mean(results,axis=2),origin='lower')
	fig.colorbar(im)
	ax.set_xlabel('Tree Sizes')
	ax.set_xticks(np.arange(len(tree_sizes)))
	ax.set_xticklabels(tree_sizes)
	ax.set_ylabel('Betas')
	ax.set_yticks(np.arange(len(betas)))
	ax.set_yticklabels(betas)
	plt.savefig("test_regression.png")
