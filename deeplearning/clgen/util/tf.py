"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os
import re

from deeplearning.clgen.util import gpu

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "tf_print_deprecation",
  False,
  "Print tensorflow deprecation warnings"
)

flags.DEFINE_boolean(
  "tf_gpu_allow_growth",
  True,
  "Force tensorflow to allocate only needed space and not full GPU memory"
)

flags.DEFINE_integer(
  "tf_logging_level",
  3,
  "Logging level of tensorflow logger"
)

flags.DEFINE_boolean(
  "tf_disable_eager",
  True,
  "Select to enable or disable eager execution. As of now, all modules use graph mode, \
  therefore eager execution must be disabled."
)

flags.DEFINE_string(
  "tf_device",
  "gpu",
  "Select device to deploy application. Valid options are 'cpu', 'gpu' and 'tpu'. [Default]: 'gpu'"
  "If GPU unavailable, it rolls back to CPU."
)

import tensorflow
tf = tensorflow

def initTensorflow():
  from eupy.native import logger as l

  tensorflow.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = FLAGS.tf_print_deprecation
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(FLAGS.tf_logging_level).lower()
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = str(FLAGS.tf_gpu_allow_growth).lower()
  available_gpus = gpu.getGPUID()
  try:
    if FLAGS.tf_device == "tpu":
      raise NotImplementedError
    elif FLAGS.tf_device == "gpu" and available_gpus is not None:
      l.getLogger().info("Selected GPU:{} {}".format(available_gpus[0]['id'], available_gpus[0]['gpu_name']))
      os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[0])
    elif re.search("gpu:[0-9]", FLAGS.tf_device) and available_gpus is not None:
      gpuid = int(FLAGS.tf_device.split(':')[-1])
      selected_gpu = None
      for gp in available_gpus:
        if int(gp['id']) == gpuid:
          selected_gpu = gp
      if selected_gpu is None:
        raise ValueError("Invalid GPU ID: {}".format(gpuid))
      l.getLogger().info("Selected GPU:{} {}".format(selected_gpu['id'], selected_gpu['gpu_name']))
      os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
    else:
      l.getLogger().info("Selected CPU device.")
      os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  except RuntimeError as e:
    raise e

  if FLAGS.tf_disable_eager:
    tensorflow.compat.v1.disable_eager_execution()
  tf = tensorflow
