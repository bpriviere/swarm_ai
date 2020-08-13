# import argparse
import yaml
import os
import tempfile
import subprocess
import numpy as np
import time
import matplotlib.pyplot as plt 

def run(data, useGLAS):
	with tempfile.TemporaryDirectory() as tmpdirname:
		input_file = os.path.join(tmpdirname, "config.yaml")

		with open(input_file, 'w') as f:
			yaml.dump(data, f)

		output_file = os.path.join(tmpdirname, "output.csv")
		start = time.time()
		if useGLAS:
			subprocess.run("buildRelease/swarmgame -i {} -n nn.yaml -o {}".format(input_file, output_file), shell=True)
		else:
			subprocess.run("buildRelease/swarmgame -i {} -o {}".format(input_file, output_file), shell=True)
		elapsed = (time.time() - start)
		data = np.loadtxt(output_file, delimiter=',', skiprows=1, ndmin=2, dtype=np.float32)
		T = data.shape[0]
		last_reward = data[-1,-1]
		return T, last_reward, elapsed

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument("input_a", help="pt file with the pyTorch state dict")
	# parser.add_argument("input_b", help="pt file with the pyTorch state dict")
	# parser.add_argument("output", help="yaml file for the weights")
	# args = parser.parse_args()

	# load original config
	with open('config.yaml') as f:
		data = yaml.load(f, Loader=yaml.FullLoader)

	all_results = []
	for tree_size in [10, 100, 1000, 10000, 100000]:
		data["tree_size"] = tree_size

		for useGLAS in [False]:#, True]:

			results = []
			for seed in range(1,11):
				data["seed"] = seed
				results.append(run(data, useGLAS))
			results = np.array(results)
			all_results.append(results)

	fig, axs = plt.subplots(3)
	axs[0].boxplot([r[:,0] for r in all_results])
	axs[0].set_ylabel("T")

	axs[1].boxplot([r[:,1] for r in all_results])
	axs[1].set_ylabel("Final reward")

	axs[2].boxplot([r[:,2] for r in all_results])
	axs[2].set_ylabel("Runtime")

	plt.show()

	# with open(args.output, 'w') as f:




# def run_mcts_batch(param, instance_key, datadir): 

# 	with tempfile.TemporaryDirectory() as tmpdirname:
# 		input_file = tmpdirname + "/config.yaml" 
# 		dh.write_mcts_config_file(param, input_file)
# 		output_file = tmpdirname + "/output.csv"
# 		print('running instance {}'.format(instance_key))
# 		subprocess.run("../mcts/cpp/buildRelease/swarmgame -i {} -o {}".format(input_file, output_file), shell=True)
# 		data = np.loadtxt(output_file, delimiter=',', skiprows=1, ndmin=2, dtype=np.float32)
# 		sim_result = dh.convert_cpp_data_to_sim_result(data,param)

# 	print('writing instance {}... '.format(instance_key))
# 	dh.write_sim_result(sim_result,datadir + instance_key)
# 	print('completed instance {}'.format(instance_key))