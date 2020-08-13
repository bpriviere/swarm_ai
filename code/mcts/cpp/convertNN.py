# import numpy as np
import torch
import argparse
import yaml

def load(file):
	state_dict = torch.load(file)
	output_dict = dict()

	for name, tensor in state_dict.items():
		output_dict[name] = tensor.numpy().tolist()
	return output_dict

def convertNN(input_a, input_b, output):
	output_dict = dict()
	output_dict["team_a"] = load(input_a)
	output_dict["team_b"] = load(input_b)

	with open(output, 'w') as f:
		yaml.dump(output_dict, f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_a", help="pt file with the pyTorch state dict")
	parser.add_argument("input_b", help="pt file with the pyTorch state dict")
	parser.add_argument("output", help="yaml file for the weights")
	args = parser.parse_args()

	convertNN(args.input_a, args.input_b, args.output)
