import progressbar
import six 

from deeplearning.clgen.tf import tf
from eupy.native import logger as l


class tfProgressBar(tf.compat.v1.train.SessionRunHook):
  """Real time progressbar to capture tf Estimator training or validation"""

  def __init__(self, 
               max_length: int,
               tensors: dict = None,
               log_steps: int = None,
               at_end: bool = False,
               is_training: bool = True,
               ):
    """
    Initialize Progress Bar Hook
    This hook shows a progress bar in output and prints after N steps tensor values provided.

	Args:
		max_length: This is the maximum threshold of the progress bar
		tensors: Optional string to tf.Tensor dictionary for the tensor values desired to be monitored, if set.
		log_steps: If set, logs tensor values once every defined number of estimator steps
		at_end: If set, prints tensor values at end of session
		is_training: If hooks is used for training or evaluation
    """
    self.max_length = max_length
    self.tensors = tensors
    self.log_steps = log_steps
    self.at_end = at_end
    self.is_training = is_training

    self.global_step = tf.compat.v1.train.get_or_create_global_step()

    if self.is_training:
      self.step_tensor = { self.global_step: self.global_step }
      self._current_epoch = 0

    if self.tensors is not None:

      only_log_at_end = False
      if self.log_steps is None:
        if self.at_end:
          only_log_at_end = True
        else:
          raise ValueError("Tensors is not None, but log_steps has not been set!")

      if not isinstance(self.tensors, dict):
        self._tag_order = self.tensors
        self.tensors = {item: item for item in self.tensors}
      else:
        self._tag_order = sorted(self.tensors.keys())

      self._timer = (
          tf.compat.v1.train.SecondOrStepTimer(every_steps=max_length) if only_log_at_end
          else tf.compat.v1.train.SecondOrStepTimer(every_steps=log_steps))

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
          tag: self._as_graph_element(tensor)
          for (tag, tensor) in self.step_tensor.items()
          }
    
    if self.tensors is not None:
      self._timer.reset()
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
