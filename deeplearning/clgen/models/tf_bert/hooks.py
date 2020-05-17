import progressbar
import six 

from deeplearning.clgen.tf import tf
from eupy.native import logger as l


class tfProgressBar(tf.compat.v1.train.SessionRunHook):
  """Real time progressbar to capture tf Estimator training or validation"""

  def __init__(self, 
               max_length: int,
               tensors = None,
               log_steps = None,
               at_end = None
               ):
    """
    Set class variables
    :param max_length:
        Upper threshold of progressbar
    """
    self.max_length = max_length
    self.log_steps = log_steps
    self.at_end = at_end
    self.tensors = tensors

    if tensors is not None:

      only_log_at_end = False
      if self.log_steps is None:
        if self.at_end is None:
          raise ValueError("Tensors is not None, but log_steps has not been set!")
        else:
          only_log_at_end = True

      if not isinstance(tensors, dict):
        self._tag_order = tensors
        tensors = {item: item for item in tensors}
      else:
        self._tag_order = sorted(tensors.keys())

      self._timer = (
          NeverTriggerTimer() if only_log_at_end
          else tf.compat.v1.train.SecondOrStepTimer(every_steps=log_steps))

  def begin(self):
    """
        Called once at initialization stage
    :param session:
        Tensorflow session
    :param coord:
        unused
    """
    self._current_step = 0
    self._timer.reset()

    self.bar = progressbar.ProgressBar(max_value = self.max_length)

    if self.tensors is not None:
      self._current_tensors = {
          tag: self._as_graph_element(tensor)
          for (tag, tensor) in self.tensors.items()
      }

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    if self.tensors is not None:
      self._should_trigger = self._timer.should_trigger_for_step(self._current_step)
      if self._should_trigger:
        return tf.estimator.SessionRunArgs(self._current_tensors)
    return None

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    self.bar.update(self._current_step)

    if self.tensors is not None:
      _ = run_context
      if self._should_trigger:
        self._log_tensors(run_values.results)

    self._current_step += 1

  def end(self, session):
    """
      Called at the end of session
    """
    self._current_step = 0
    if self.at_end:
      values = session.run(self._current_tensors)
      self._log_tensors(values)

  def _log_tensors(self, tensor_values):

    elapsed_secs, _ = self._timer.update_last_triggered_step(self._current_step)
    stats = []

    for tag in self._tag_order:
      stats.append("{}: {}".format(tag, tensor_values[tag]))
    if elapsed_secs is not None:
      l.getLogger().info("Epoch {} ({:3f} sec)".format(", ".join(stats), elapsed_secs))
    else:
      l.getLogger().info("Epoch {}".format(", ".join(stats)))

  def _as_graph_element(self, obj):
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

class _HookTimer(object):
  """Base timer for determining when Hooks should trigger.
  Should not be instantiated directly.
  """

  def __init__(self):
    pass

  def reset(self):
    """Resets the timer."""
    pass

  def should_trigger_for_step(self, step):
    """Return true if the timer should trigger for the specified step."""
    raise NotImplementedError

  def update_last_triggered_step(self, step):
    """Update the last triggered time and step number.
    Args:
      step: The current step.
    Returns:
      A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
      of seconds between the current trigger and the last one (a float), and
      `elapsed_steps` is the number of steps between the current trigger and
      the last one. Both values will be set to `None` on the first trigger.
    """
    raise NotImplementedError

  def last_triggered_step(self):
    """Returns the last triggered time step or None if never triggered."""
    raise NotImplementedError

class NeverTriggerTimer(_HookTimer):
  """Timer that never triggers."""

  def should_trigger_for_step(self, step):
    _ = step
    return False

  def update_last_triggered_step(self, step):
    _ = step
    return (None, None)

  def last_triggered_step(self):
    return None
