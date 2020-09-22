import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from itertools import repeat

class DeepSetDataset(torch.utils.data.Dataset):
  def __init__(self, data_dict):
    self.data_dict = data_dict

  def __getitem__(self, index):
    key, idx = index
    return self.data_dict[key][idx]

  def __len__(self):
    size = 0
    for _, tensor in self.data_dict.items():
      size += tensor.size(0)
    return size


class DeepSetSampler(torch.utils.data.Sampler):
  def __init__(self, dataset, batch_size, drop_last):
    self.dataset = dataset
    self.batch_size = batch_size
    self.drop_last = drop_last
    
    # compute size
    self.size = 0
    self.batch_samplers = []
    for key, tensor in dataset.data_dict.items():
      # random_sampler = torch.utils.data.RandomSampler(tensor)
      random_sampler = torch.utils.data.SubsetRandomSampler(list(zip(repeat(key), range(len(tensor)))))
      batch_sampler = torch.utils.data.BatchSampler(random_sampler, batch_size, drop_last)
      self.size += len(batch_sampler)
      self.batch_samplers.append(batch_sampler)

  def __iter__(self):
    iterables = [iter(s) for s in self.batch_samplers]
    while len(iterables) > 0:
      random.shuffle(iterables)
      try:
        yield next(iterables[-1])
      except StopIteration:
        iterables.pop()

  def __len__(self):
    return self.size


if __name__ == '__main__':

  data_type1 = torch.zeros((10,2))
  data_type2 = torch.zeros((5,4))

  dataset = DeepSetDataset({'0a_1b': data_type1, '1a_0b': data_type2})

  # sampler = DeepSetSampler(dataset, 2, True)
  # print(len(sampler))
  # for s in sampler:
  #   print(s)

  sampler = DeepSetSampler(dataset, 2, False)
  loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

  for data in loader:
    print(data)
