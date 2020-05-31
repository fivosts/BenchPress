"""A wrapper module to include tensorflow with some options"""
from absl import flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "tf_print_deprecation",
  False,
  "Print tensorflow deprecation warnings"
)

flags.DEFINE_string(
  "tf_logging_level",
  '3',
  "Logging level of tensorflow logger"
)

flags.DEFINE_boolean(
  "tf_disable_eager",
  True,
  "Select to enable or disable eager execution. As of now, all modules use graph mode, ",
  "therefore eager execution must be disabled."
)

tf = None
if tf is None:
  import tensorflow
  tf = tensorflow
  tf.python.util.deprecation._PRINT_DEPRECATION_WARNINGS = FLAGS.tf_print_deprecation
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tf_logging_level
  if FLAGS.tf_disable_eager:
    tf.compat.v1.disable_eager_execution()
