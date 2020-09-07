"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "pt_cpu_only",
  False,
  "Do not use GPU/TPU in pytorch session."
)

import torch

try:
  import torch_xla.core.xla_model
  import torch_xla.debug.metrics
  import torch_xla.distributed.parallel_loader
  torch_xla     = torch_xla.core.xla_model
  torch_xla_met = torch_xla.debug.metrics
  torch_ploader = torch_xla.distributed.parallel_loader
  torch_tpu_available = True
except ImportError:
  torch_tpu_available = False

def initPytorch(local_rank = -1):
  global torch_tpu_available
  if FLAGS.pt_cpu_only:
    device = torch.device("cpu")
    num_gpus = 0
  elif torch_tpu_available:
    device = torch_xla.xla_device()
    num_gpus = 0
  elif local_rank == -1:
    # if num_gpus is > 1 we'll use nn.DataParallel.
    # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
    # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
    # trigger an error that a device index is missing. Index 0 takes into account the
    # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
    # will use the first GPU in that env, i.e. GPU#1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
  else:
    # Here, we'll use torch.distributed.
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    num_gpus = 1

  if device.type == "cuda":
    torch.cuda.set_device(device)
  return device, num_gpus
device, num_gpus = initPytorch()
