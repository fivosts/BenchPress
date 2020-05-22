import progressbar
import six 

from deeplearning.clgen.tf import tf
from eupy.native import logger as l
"""
All hooks deployed for this implementation of BERT.
These hooks must be strictly called within model_fn function
and be passed to EstimatorSpec.
"""

def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = tf.python.framework.ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element


class tfEstimatorHooks(tf.compat.v1.train.SessionRunHook):
  """Base class for Estimator Hooks, used for this BERT model"""
  def __init__(self,
              mode: tf.compat.v1.estimator.ModeKeys,
              ):
    """
    Base class hook initialization
  Args:
    mode: If hooks is used for training or evaluation
    """

    self.global_step = tf.compat.v1.train.get_or_create_global_step()
    self.current_step = None

    if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
      ## Training
      self.is_training = True
    elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:
      ## Validation
      self.is_training = False
    else:
      ## Sampling
      self.is_training = False

    return

    def begin(self):
      if self.is_training:
        self.step_tensor = {
            self.global_step: _as_graph_element(self.global_step)
          }
      return

    def before_run(self, run_context):
      return tf.estimator.SessionRunArgs(self.step_tensor)

    def after_run(self, run_context, run_values):
      self._current_step = run_values.results[self.global_step]
      return

    def end(self, session):
      return

class tfProgressBar(tfEstimatorHooks):
  """Real time progressbar to capture tf Estimator training or validation"""

  def __init__(self, 
               max_length: int,
               mode: tf.compat.v1.estimator.ModeKeys = None,
               ):
    """
    Initialize Progress Bar Hook
    This hook shows a progress bar in output and prints after N steps tensor values provided.

  Args:
    max_length: This is the maximum threshold of the progress bar
    tensors: Optional string to tf.Tensor dictionary for the tensor values desired to be monitored, if set.
    log_steps: If set, logs tensor values once every defined number of estimator steps
    at_end: If set, prints tensor values at end of session
    mode: If hooks is used for training or evaluation
    """
    super(tfProgressBar, self).__init__(mode)

    self.max_length = max_length

    if self.is_training:
      self.step_tensor = { self.global_step: self.global_step }
      self._current_epoch = 0

  def begin(self):
    """
        Called once at initialization stage
    :param session:
        Tensorflow session
    :param coord:
        unused
    """
    self._trigger_step = 0
    self.bar = progressbar.ProgressBar(max_value = self.max_length)

    if self.is_training:

      self.step_tensor = {
          tag: _as_graph_element(tensor)
          for (tag, tensor) in self.step_tensor.items()
          }
    
    if self.tensors is not None:
      self._timer.reset()
      self._current_tensors = {
          tag: _as_graph_element(tensor)
          for (tag, tensor) in self.tensors.items()
      }

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    if self.tensors is not None:
      if self._timer.should_trigger_for_step(self._trigger_step):
        if self.is_training:
          self.session_dict = {
            'step_tensor': 0,
            'value_tensor': 1,
          }
          return tf.estimator.SessionRunArgs([self.step_tensor, self._current_tensors])
        else:
          self.session_dict = {
            'value_tensor': 0,
          }
          return tf.estimator.SessionRunArgs([self._current_tensors])

    if self.is_training:
      self.session_dict = {
        'step_tensor': 0,
      }
      return tf.estimator.SessionRunArgs([self.step_tensor])
    else:
      self.session_dict = {}
      return None

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    ##  0th element is always global_step, see how list is ordered in SessionRunArgs

    if 'step_tensor' in self.session_dict:
      self._current_step = run_values.results[self.session_dict['step_tensor']][self.global_step]
      self._current_epoch = int(self._current_step / self.log_steps)
    else:
      self._current_step = self._trigger_step

    self.bar.update(self._current_step) 

    _ = run_context
    if 'value_tensor' in self.session_dict:
      self._log_tensors(run_values.results[self.session_dict['value_tensor']])

    self._trigger_step += 1

  def end(self, session):
    """
      Called at the end of session
    """
    if self.tensors is not None and self.at_end:
      values = session.run(self._current_tensors)
      self._log_tensors(values)

  def _log_tensors(self, tensor_values):

    elapsed_secs, _ = self._timer.update_last_triggered_step(self._trigger_step)
    stats = []

    for tag in self._tag_order:
      stats.append("{}: {:.5f}".format(tag, tensor_values[tag]))
    if elapsed_secs is not None:
      l.getLogger().info("Epoch {} {} - {:.3f} sec".format(self._current_epoch, ", ".join(stats), elapsed_secs))
    elif self._current_epoch > 0:
      l.getLogger().info("Epoch {} {}".format(self._current_epoch, ", ".join(stats)))
    else:
      l.getLogger().info("Initialization: {}".format(", ".join(stats)))

class tfLogTensorHook(tfEstimatorHooks):

  def __init__(self,
               tensors: dict,
               log_steps: int = None,
               at_end: bool = False,
              ):
    super(tfLogTensorHook, self).__init__(mode)

    self.tensors = tensors
    self.log_steps = log_steps
    self.at_end = at_end

    if log_steps is None and not at_end:
      raise ValueError("Neither log_steps nor at_end have been set. Select at least one.")

    self.timer = tf.compat.v1.train.SecondOrStepTimer(
      every_steps = (max_length if log_steps is None else log_steps)
      )

    self.tensor_tags = sorted(self.tensors.keys())
    return
