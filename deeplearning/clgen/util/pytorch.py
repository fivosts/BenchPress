"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os

from deeplearning.clgen.util import gpu
from deeplearning.clgen.util import environment

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "pt_cpu_only",
  False,
  "Do not use GPU/TPU in pytorch session."
)

flags.DEFINE_string(
  "DDP_backend",
  "nccl",
  "Select backend for Distributed Data Parallel. Valid choices are \'nccl\' and \'gloo\'"
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

offset_device = None
devices       = None
device        = None
num_gpus      = None
num_nodes     = None

def initPytorch() -> None:
  global torch_tpu_available
  global offset_device
  global devices
  global device
  global num_gpus
  global num_nodes
  if FLAGS.pt_cpu_only:
    device = torch.device("cpu")
    num_gpus = 0
  elif torch_tpu_available:
    device = torch_xla.xla_device()
    num_gpus = 0
  elif environment.WORLD_SIZE == 1:
    # if num_gpus is > 1 we'll use nn.DataParallel.
    # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
    # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
    # trigger an error that a device index is missing. Index 0 takes into account the
    # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
    # will use the first GPU in that env, i.e. GPU#1
    offset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      available_gpus = gpu.getGPUID()
      devices = ["cuda:{}".format(str(x['id'])) for x in available_gpus]
    device         = torch.device(
      "cuda:{}".format(str(available_gpus[0]['id'])) if torch.cuda.is_available() and available_gpus else "cpu"
    )
    num_gpus = torch.cuda.device_count()
    if device.type == "cuda":
      torch.cuda.set_device(device)
  else:
    # Here, we'll use torch.distributed.
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs.
    # This branch will trigger DistributedDataParalel instead of simple DP.
    # Distributed training prohibits manual selection of GPUs and takes for granted that cuda is available.
    tcp_store = torch.distributed.TCPStore(
      environment.MASTER_ADDR,
      environment.MASTER_PORT,
      environment.WORLD_SIZE,
      environment.WORLD_RANK == 0
    )
    torch.distributed.init_process_group(
      backend    = FLAGS.DDP_backend,
      store      = tcp_store,
      rank       = environment.WORLD_RANK,
      world_size = environment.WORLD_SIZE,
    )
    num_nodes = torch.distributed.get_world_size()
    num_gpus  = num_nodes # TODO ?

    device = torch.device("cuda", environment.LOCAL_RANK)
    offset_device = torch.device("cuda", environment.LOCAL_RANK)
    
    torch.cuda.set_device(device)

  return