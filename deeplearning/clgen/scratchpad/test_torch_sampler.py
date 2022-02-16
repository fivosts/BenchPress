"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import torch

WORLD_SIZE = 16

dataset_tensor = [0, 1, 2, 3]
cumulative_sizes = [100000, 200000, 300000, 400000]
dset_iter = iter(dataset_tensor)

def get_rand_tensor(epoch, dset_idx, world_rank):
  # global dataset_tensor
  # global dset_iter
  # try:
  #   dataset_idx = next(dset_iter)
  # except StopIteration:
  #   dset_iter = iter(dataset_tensor)
  #   dataset_idx = next(dset_iter)
  dataset_idx = dset_idx
  lb, ub = cumulative_sizes[dataset_idx - 1] if dataset_idx else 0, cumulative_sizes[dataset_idx]
  bounds = (lb, ub)

  generator = torch.Generator()
  generator.manual_seed(epoch)

  size = bounds[1] - bounds[0]
  rand_tensor = [x + bounds[0] for x in torch.randperm(bounds[1] - bounds[0], generator = generator).tolist()]


  rounded_total = (len(rand_tensor) // WORLD_SIZE) * WORLD_SIZE
  # print(rounded_total, rand_tensor, world_rank, rounded_total, WORLD_SIZE)
  rand_tensor   = rand_tensor[world_rank:rounded_total:WORLD_SIZE]
  return rand_tensor

for y in range(20):
  idx = y
  l1, l2, l3, l4 = get_rand_tensor(0, idx%4, 0), get_rand_tensor(0, idx%4, 1), get_rand_tensor(0, idx%4, 2), get_rand_tensor(0, idx%4, 3)

  visited = set()
  for x in l1 + l2 + l3 + l4:
    if x in visited:
      print(visited)
      print(x)
      raise ValueError("Ton ipiame!")
    else:
      visited.add(x)
  print("Ok")
