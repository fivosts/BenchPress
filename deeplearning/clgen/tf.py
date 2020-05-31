"""A wrapper module to include tensorflow with some options"""
# from absl import flags
import os
## TODO find a way to init absl flags
# FLAGS = flags.FLAGS

# flags.DEFINE_boolean(
#   "tf_print_deprecation",
#   False,
#   "Print tensorflow deprecation warnings"
# )

# flags.DEFINE_string(
#   "tf_logging_level",
#   '3',
#   "Logging level of tensorflow logger"
# )

# flags.DEFINE_boolean(
#   "tf_disable_eager",
#   True,
#   "Select to enable or disable eager execution. As of now, all modules use graph mode, \
#   therefore eager execution must be disabled."
# )

tf = None
if tf is None:
  import tensorflow
  tensorflow.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = False
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  tensorflow.compat.v1.disable_eager_execution()
  tf = tensorflow
