import progressbar

from deeplearning.clgen.tf import tf
from eupy.native import logger as l


class tfProgressBar(tf.compat.v1.train.SessionRunHook):
  """Real time progressbar to capture tf Estimator training or validation"""

  def __init__(self, length: int):
    """
    Set class variables
    :param length:
        Upper threshold of progressbar
    """
    self.max_length = length

  def begin(self):
    """
        Called once at initialization stage
    :param session:
        Tensorflow session
    :param coord:
        unused
    """
    self.current_step = 0
    self.bar = progressbar.ProgressBar(max_value = self.max_length)

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    self.current_step += 1
    return None

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    self.bar.update(self.current_step)

  def end(self, session):
    """
      Called at the end of session
    """
    self.current_step = 0