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


class _tfEstimatorHooks(tf.compat.v1.train.SessionRunHook):
  """Base class for Estimator Hooks, used for this BERT model"""
  def __init__(self,
              mode: tf.compat.v1.estimator.ModeKeys,
              ):
    """
    Base class hook initialization
  Args:
    mode: If hooks is used for training or evaluation
    """

    if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
      ## Training
      self.is_training = True
      self.current_step = None
    elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:
      ## Validation
      self.is_training = False
      self.current_step = 0
    elif mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
      ## Sampling
      self.is_training = False
      self.current_step = None
    else:
      raise ValueError("mode for hook has not been provided")

    self.global_step = _as_graph_element(tf.compat.v1.train.get_or_create_global_step())
    return

  def begin(self):
    return

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._buildSessionDict())

  def after_run(self, run_context, run_values):
    if self.is_training:
      self.current_step = run_values.results[self.global_step]
    else:
      self.current_step += 1
    return

  def end(self, session):
    return

  def _buildSessionDict(self,
                        session: dict = {}
                        ) -> dict:
    """
    Builds dictionary passed to SessionRunArgs on top of default KV pairs of base class
    Args: 
      session: dictionary mapping key->tensor
    Returns:
      dictionary 
    """
    if not isinstance(session, dict):
      raise ValueError("session dict is not in dict format")
    if self.global_step in session:
      raise KeyError("global_step tensor has been set by derived class, while base class should do that.")
    session[self.global_step] = self.global_step
    return session

class tfProgressBar(_tfEstimatorHooks):
  """Real time progressbar to capture tf Estimator training or validation"""

  def __init__(self, 
               max_length: int,
               mode: tf.compat.v1.estimator.ModeKeys = tf.compat.v1.estimator.ModeKeys.TRAIN,
               ):
    """
    Initialize Progress Bar Hook
    This hook shows a progress bar in output and prints after N steps tensor values provided.

    Args:
      max_length: This is the maximum threshold of the progress bar
      mode: If hooks is used for training or evaluation
    """
    super(tfProgressBar, self).__init__(mode)
    self.max_length = max_length

  def begin(self):
    """
    Called once at initialization stage
    """
    self.bar = progressbar.ProgressBar(max_value = self.max_length)
    return

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    super(tfProgressBar, self).after_run(run_context, run_values)
    self.bar.update(self.current_step) 
    return

class tfLogTensorHook(_tfEstimatorHooks):

  def __init__(self,
               tensors: dict,
               log_steps: int = None,
               at_end: bool = False,
               mode: tf.compat.v1.estimator.ModeKeys = tf.compat.v1.estimator.ModeKeys.TRAIN,
              ):
    """
    Args:
      tensors: Optional string to tf.Tensor dictionary for the tensor values desired to be monitored, if set.
      log_steps: If set, logs tensor values once every defined number of estimator steps
      at_end: If set, prints tensor values at end of session
      mode: If hooks is used for training or evaluation
    """
    super(tfLogTensorHook, self).__init__(mode)

    self.tensor_tags = sorted(tensors.keys())
    self.tensors = {
        tag: _as_graph_element(tensor)
        for (tag, tensor) in tensors.items()
    }
    self.log_steps = log_steps
    self.at_end = at_end
    self.step_triggered = False

    if log_steps is None and not at_end:
      raise ValueError("Neither log_steps nor at_end have been set. Select at least one.")

    self.timer = tf.compat.v1.train.SecondOrStepTimer(
      every_steps = (max_length if log_steps is None else log_steps)
      )

    return

  def begin(self):
    """
        Called once at initialization stage
    """
    self.trigger_step = 0
    self.current_epoch = 0
    self.timer.reset()
    return

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """

    if self.timer.should_trigger_for_step(self.trigger_step):
      self.step_triggered = True
      return tf.estimator.SessionRunArgs(self._buildSessionDict({'tensors' : self.tensors}))
    else:
      self.step_triggered = False
      return super(tfLogTensorHook, self).before_run(run_context)
    
    return None

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    ##  0th element is always global_step, see how list is ordered in SessionRunArgs
    super(tfLogTensorHook, self).after_run(run_context, run_values)
    self.current_epoch = int(self.current_step / self.log_steps)

    if self.step_triggered:
      self._log_tensors(run_values.results['tensors'])

    self.trigger_step += 1

  def end(self, session):
    """
      Called at the end of session
    """
    super(tfLogTensorHook, self).end(session)
    if self.at_end:
      end_values = session.run(self.tensors)
      self._log_tensors(end_values)

  def _log_tensors(self, tensor_values):

    if self.is_training:
      elapsed_secs, _ = self.timer.update_last_triggered_step(self.trigger_step)
    else:
      elapsed_secs = None

    stats = []
    for tag in self.tensor_tags:
      stats.append("{}: {:.5f}".format(tag, tensor_values[tag]))
    if elapsed_secs is not None:
      l.getLogger().info("Epoch {} {} - {:.3f} sec".format(self.current_epoch, ", ".join(stats), elapsed_secs))
    elif self.current_epoch > 0:
      l.getLogger().info("Epoch {} {}".format(self.current_epoch, ", ".join(stats)))
    else:
      if self.is_training:
        l.getLogger().info("Initialization: {}".format(", ".join(stats)))
      else:
        l.getLogger().info("Tensor Values: {}".format(", ".join(stats)))
