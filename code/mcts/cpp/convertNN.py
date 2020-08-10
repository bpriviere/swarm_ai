# import numpy as np
import torch
import argparse
import yaml


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="pt file with the pyTorch state dict")
	parser.add_argument("output", help="yaml file for the weights")
	args = parser.parse_args()

	state_dict = torch.load(args.input)

	output_dict = dict()

	for name, tensor in state_dict.items():
		output_dict[name] = tensor.numpy().tolist()


	with open(args.output, 'w') as f:
		yaml.dump(output_dict, f)
