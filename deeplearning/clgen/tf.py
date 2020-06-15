"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os
## TODO find a way to init absl flags
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
  1,
  "Logging level of tensorflow logger"
)

flags.DEFINE_boolean(
  "tf_disable_eager",
  True,
  "Select to enable or disable eager execution. As of now, all modules use graph mode, \
  therefore eager execution must be disabled."
)

import tensorflow
tf = tensorflow

def initTensorflow():
  tensorflow.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = FLAGS.tf_print_deprecation
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(FLAGS.tf_logging_level).lower()
  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = str(FLAGS.tf_gpu_allow_growth)
  if FLAGS.tf_disable_eager:
    tensorflow.compat.v1.disable_eager_execution()
  tf = tensorflow
