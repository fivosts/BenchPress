import progressbar
import six 
import humanize
import numpy as np
import glob
import pathlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from deeplearning.clgen.util.tf import tf
from deeplearning.clgen import validation_database
from eupy.native import logger as l
from eupy.native import plotter as plt

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
    self.session_dict = {}
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
    """
      Initialize the session dictionary for the base class
      session_dict will be incremented by derived classes that
      need extra tensors to be evaluated
    """
    self.session_dict = {
      self.global_step: self.global_step
    }
    return

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self.session_dict)

  def after_run(self, run_context, run_values):
    if self.is_training:
      self.current_step = run_values.results[self.global_step]
    else:
      self.current_step += 1
    return

  def end(self, session):
    return

class AverageSummarySaverHook(_tfEstimatorHooks):
  """
    Similar functionality to SummarySaverHook that
    stores averaged tensors instead of step-instant values.
  """
  def __init__(self,
               tensors: dict,
               save_steps: int,
               output_dir: str,
               show_average: bool = True,
               mode: tf.compat.v1.estimator.ModeKeys = tf.compat.v1.estimator.ModeKeys.TRAIN,
              ):
    """
    Args:
      tensors: Optional string to tf.Tensor dictionary for the tensor values desired to be monitored, if set.
      save_steps: If set, logs tensor values once every defined number of estimator steps
      output_dir: Location of tf.event summary files.
      mode: If hooks is used for training or evaluation
    """
    super(AverageSummarySaverHook, self).__init__(mode)

    self.tensors = {
      summary_tensor.name.replace(":0", ""): tensor
        for (summary_tensor, tensor) in zip(tensors[0], tensors[1])
    }
    self.result = {k: [] for k in self.tensors}

    self.save_steps     = save_steps
    self.step_triggered = False
    self.show_average   = show_average
    self.output_dir     = output_dir

    self.timer = tf.compat.v1.train.SecondOrStepTimer(every_steps = save_steps)
    return

  def begin(self):
    """
        Called once at initialization stage
    """
    super(AverageSummarySaverHook, self).begin()
    self.summary_writer = tf.python.training.summary_io.SummaryWriterCache.get(self.output_dir)
    self.trigger_step = 0
    self.session_dict['tensors'] = self.tensors
    self.timer.reset()
    return

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    self.step_triggered = self.timer.should_trigger_for_step(self.trigger_step)
    return tf.estimator.SessionRunArgs(self.session_dict)

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    super(AverageSummarySaverHook, self).after_run(run_context, run_values)

    for tag in self.tensors:
      if self.show_average:
        self.result[tag].append(run_values.results['tensors'][tag])
      else:
        self.result[tag] = [run_values.results['tensors'][tag]]

    if self.trigger_step == 0:
      self.summary_writer.add_session_log(
        tf.core.util.event_pb2.SessionLog(status=tf.core.util.event_pb2.SessionLog.START),
        self.current_step
      )

    if self.step_triggered:
      self.result = { k: (sum(v) / len(v)) for (k, v) in self.result.items() }
      self._save_summary(self.result)
      self.result = {k: [] for k in self.result}

    self.trigger_step += 1

  def _save_summary(self, tensor_values):

    if self.is_training:
      elapsed_secs, _ = self.timer.update_last_triggered_step(self.trigger_step)
    else:
      elapsed_secs = None

    tensor_summary = []
    for (key, value) in tensor_values.items():
      tensor_summary.append(
        tf.core.framework.summary_pb2.Summary.Value(
          tag = key, simple_value = value
        )
      )
    summary = tf.core.framework.summary_pb2.Summary(value = tensor_summary)
    self.summary_writer.add_summary(summary, self.current_step)

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
    super(tfProgressBar, self).begin()
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
               log_steps: int,
               show_average: bool = True,
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
    self.result = {k: [] for k in self.tensor_tags}

    self.log_steps      = log_steps
    self.at_end         = at_end
    self.step_triggered = False
    self.show_average   = show_average

    self.timer = tf.compat.v1.train.SecondOrStepTimer(every_steps = log_steps)
    return

  def begin(self):
    """
        Called once at initialization stage
    """
    super(tfLogTensorHook, self).begin()
    self.trigger_step = 0
    self.current_epoch = 0
    self.session_dict['tensors'] = self.tensors
    self.timer.reset()
    return

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    self.step_triggered = self.timer.should_trigger_for_step(self.trigger_step)
    return tf.estimator.SessionRunArgs(self.session_dict)
    
  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    super(tfLogTensorHook, self).after_run(run_context, run_values)
    self.current_epoch = int(self.current_step / self.log_steps)

    for tag in self.tensor_tags:
      if self.show_average:
        self.result[tag].append(run_values.results['tensors'][tag])
      else:
        self.result[tag] = [run_values.results['tensors'][tag]]

    if self.step_triggered:
      self.result = { k: (sum(v) / len(v)) for (k, v) in self.result.items() }
      self._log_tensors(self.result)
      self.result = {k: [] for k in self.result}

    self.trigger_step += 1

  def end(self, session):
    """
      Called at the end of session
    """
    super(tfLogTensorHook, self).end(session)
    if self.at_end:
      end_values = session.run(self.tensors)
      for tag in self.tensor_tags:
        if self.show_average:
          self.result[tag].append(end_values[tag])
        else:
          self.results[tag] = [end_values[tag]]

      self.result = { k: (sum(v) / len(v)) for (k, v) in self.result.items() }

  def _log_tensors(self, tensor_values):

    if self.is_training:
      elapsed_secs, _ = self.timer.update_last_triggered_step(self.trigger_step)
    else:
      elapsed_secs = None

    stats = []
    for tag in self.tensor_tags:
      stats.append("{}: {:.5f}".format(tag, tensor_values[tag]))
    if elapsed_secs is not None:
      l.getLogger().info("Epoch {} {} - {}".format(self.current_epoch, ", ".join(stats), humanize.naturaldelta(elapsed_secs)))
    elif self.current_epoch > 0:
      l.getLogger().info("Epoch {} {}".format(self.current_epoch, ", ".join(stats)))
    else:
      if self.is_training:
        l.getLogger().info("Initialization: {}".format(", ".join(stats)))
      else:
        l.getLogger().info("Tensor Values: {}".format(", ".join(stats)))

class tfPlotTensorHook(_tfEstimatorHooks):
  """Real time training hook that plots tensors against training step."""
  def __init__(self,
               tensors: dict,
               log_steps: int,
               output_dir: pathlib.Path,
               mode: tf.compat.v1.estimator.ModeKeys = tf.compat.v1.estimator.ModeKeys.TRAIN,
              ):
    """
    Args:
      tensors: String to tf.Tensor dictionary for the plotted values.
      log_steps: If set, logs tensor values once every defined number of estimator steps
      mode: If hooks is used for training or evaluation
    """
    if mode != tf.compat.v1.estimator.ModeKeys.TRAIN:
      raise ValueError("tfPlotTensorHook can only be used for training mode.")
    super(tfPlotTensorHook, self).__init__(mode)
    self.tensors = {
      summary_tensor.name.replace(":0", ""): tensor 
        for (summary_tensor, tensor) in zip(tensors[0], tensors[1])
    }
    self.epoch_values = {
                          tag: {'value': [], 'step': []} 
                            for tag in self.tensors
                        }
    self.results = {
                     tag: {'value': [], 'step': []}
                       for tag in self.tensors
                   }

    if len(glob.glob(str(output_dir / "events.out.tfevents*"))) != 0:
      try:
        event_acc = EventAccumulator(str(output_dir))
        event_acc.Reload()
        for k in self.tensors:
          wt, step, value = zip(*event_acc.Scalars(k))
          self.results[k] = {
            'value': list(value),
            'step' : list(step),
          }
      except KeyError:
        pass

    self.log_steps    = log_steps
    self.output_dir   = output_dir

    self.step_triggered = False
    self.timer = tf.compat.v1.train.SecondOrStepTimer(every_steps = log_steps)
    return

  def begin(self):
    """
        Called once at initialization stage
    """
    super(tfPlotTensorHook, self).begin()
    self.trigger_step = 0
    self.session_dict['tensors'] = self.tensors
    self.timer.reset()
    return

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    self.step_triggered = self.timer.should_trigger_for_step(self.trigger_step)
    return tf.estimator.SessionRunArgs(self.session_dict)
  
  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    super(tfPlotTensorHook, self).after_run(run_context, run_values)

    for tag in self.tensors:
      self.epoch_values[tag]['value'].append(run_values.results['tensors'][tag])
      self.epoch_values[tag]['step'].append(run_values.results[self.global_step])
      if self.step_triggered:
        self.results[tag]['value'].append(sum(self.epoch_values[tag]['value']) / 
                                          len(self.epoch_values[tag]['value']))
        self.results[tag]['step'].append(run_values.results[self.global_step])
        self.epoch_values[tag] = {'value': [], 'step': []}

    if self.step_triggered:
      self._plot_tensors(self.results)
    self.trigger_step += 1

  def _plot_tensors(self, tensor_values):

    _, _ = self.timer.update_last_triggered_step(self.trigger_step)
    for (key, value) in tensor_values.items():
      key_str = str(pathlib.Path(key).stem)
      plt.linesSingleAxis(
          {key_str: {'y': value['value'], 'x': value['step']}},
          y_label = (key_str, 13),
          x_label = ("Train step", 13),
          plot_title = (key_str, 20),
          x_lim = [0, value['step'][-1] + 0.01 * value['step'][-1]],
          y_lim = 1.1 * max(value['value']),
          legend = False,
          showfig = False,
          savefig = str(self.output_dir / "{}.png".format(key_str)),
          force_init = True,
        )
    return


class writeValidationDB(_tfEstimatorHooks):
  """Real time storage hook for validation results"""

  def __init__(self,
               mode,
               url,
               atomizer,
               seen_in_training,
               original_input,
               input_ids,
               input_mask,
               masked_lm_positions,
               masked_lm_ids,
               masked_lm_weights,
               masked_lm_lengths,
               next_sentence_labels,
               masked_lm_predictions,
               next_sentence_predictions,
               ):
    """
    Initialize writeValidationDB
    Stores input, target predictions, actual predictions, positions, step
    during validation to database.

    Args:
      All input and output tensors for each single validation step.
    """
    super(writeValidationDB, self).__init__(mode)

    self.atomizer                  = atomizer
    self.val_db                    = validation_database.ValidationDatabase("sqlite:///{}".format(url))
    self.val_id                    = self.val_db.count

    self.seen_in_training          = seen_in_training
    self.original_input            = original_input
    self.input_ids                 = input_ids
    self.input_mask                = input_mask
    self.masked_lm_positions       = masked_lm_positions
    self.masked_lm_ids             = masked_lm_ids
    self.masked_lm_weights         = masked_lm_weights
    self.masked_lm_lengths         = masked_lm_lengths
    self.next_sentence_labels      = next_sentence_labels
    self.masked_lm_predictions     = masked_lm_predictions
    self.next_sentence_predictions = next_sentence_predictions
    return

  def begin(self):
    """
        Called once at initialization stage
    """
    super(writeValidationDB, self).begin()
    self.session_dict[self.seen_in_training]          = self.seen_in_training
    self.session_dict[self.original_input]            = self.original_input
    self.session_dict[self.input_ids]                 = self.input_ids
    self.session_dict[self.input_mask]                = self.input_mask
    self.session_dict[self.masked_lm_positions]       = self.masked_lm_positions
    self.session_dict[self.masked_lm_ids]             = self.masked_lm_ids
    self.session_dict[self.masked_lm_weights]         = self.masked_lm_weights
    self.session_dict[self.masked_lm_lengths]         = self.masked_lm_lengths
    self.session_dict[self.next_sentence_labels]      = self.next_sentence_labels
    self.session_dict[self.masked_lm_predictions]     = self.masked_lm_predictions
    self.session_dict[self.next_sentence_predictions] = self.next_sentence_predictions
    return

  def before_run(self, run_context):
    """
      Called before session.run()
      Any tensor/op should be declared here in order to be evaluated
      returns None or SessionRunArgs()
    """
    return tf.estimator.SessionRunArgs(self.session_dict)

  def after_run(self, run_context, run_values):
    """
      Requested tensors are evaluated and their values are available
    """
    super(writeValidationDB, self).after_run(run_context, run_values)

    batch_size = run_values.results[self.input_ids].shape[0]

    masked_lm_predictions = np.reshape(
      run_values.results[self.masked_lm_predictions],
      (batch_size, int(len(run_values.results[self.masked_lm_predictions]) / batch_size))
    )
    next_sentence_predictions = np.reshape(
      run_values.results[self.next_sentence_predictions],
      (batch_size, int(len(run_values.results[self.next_sentence_predictions]) / batch_size))
    )

    assert run_values.results[self.original_input].shape[0]       == batch_size
    assert run_values.results[self.input_ids].shape[0]            == batch_size
    assert run_values.results[self.input_mask].shape[0]           == batch_size
    assert run_values.results[self.masked_lm_positions].shape[0]  == batch_size
    assert run_values.results[self.masked_lm_ids].shape[0]        == batch_size
    assert run_values.results[self.masked_lm_weights].shape[0]    == batch_size
    assert run_values.results[self.masked_lm_lengths].shape[0]    == batch_size
    assert run_values.results[self.next_sentence_labels].shape[0] == batch_size
    assert masked_lm_predictions.shape[0]                         == batch_size
    assert next_sentence_predictions.shape[0]                     == batch_size

    with self.val_db.Session(commit = True) as session:
      for b in range(batch_size):
        val_trace = validation_database.BERTtfValFile(
          **validation_database.BERTtfValFile.FromArgs(
            atomizer = self.atomizer,
            id       = self.val_id,
            train_step                = run_values.results[self.global_step],
            seen_in_training          = run_values.results[self.seen_in_training][b],
            original_input            = run_values.results[self.original_input][b],
            input_ids                 = run_values.results[self.input_ids][b],
            input_mask                = run_values.results[self.input_mask][b],
            masked_lm_positions       = run_values.results[self.masked_lm_positions][b],
            masked_lm_ids             = run_values.results[self.masked_lm_ids][b],
            masked_lm_weights         = run_values.results[self.masked_lm_weights][b],
            masked_lm_lengths         = run_values.results[self.masked_lm_lengths][b],
            next_sentence_labels      = run_values.results[self.next_sentence_labels][b],
            masked_lm_predictions     = masked_lm_predictions[b],
            next_sentence_predictions = next_sentence_predictions[b],
          )
        )
        try:
          exists = session.query(validation_database.BERTtfValFile.sha256).filter_by(sha256 = val_trace.sha256).scalar() is not None
        except sqlalchemy.orm.exc.MultipleResultsFound as e:
          l.getLogger().error("Selected sha256 has been already found more than once.")
          raise e
        if not exists:
          session.add(val_trace)
          self.val_id += 1
    return