# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""BenchPress models using a Keras backend."""
import copy
import os
import pathlib
import time
import typing
import humanize

from deeplearning.benchpress.util import logging as l

import numpy as np
import progressbar

import tensorflow_addons as tfa

from deeplearning.benchpress.samplers import samplers
from deeplearning.benchpress.models import telemetry
from deeplearning.benchpress.models import backends
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.util import tf as local_tf
from deeplearning.benchpress.models.tf_sequential.data_generator import TensorflowBatchGenerator
from absl import flags

FLAGS = flags.FLAGS

tf = local_tf.tf

flags.DEFINE_boolean(
  "clgen_tf_backend_reset_inference_state_between_batches",
  False,
  "If set, reset the network state between sample batches. Else, the model "
  "state is unaffected.",
)
flags.DEFINE_integer(
  "clgen_tf_backend_tensorboard_summary_step_count",
  25,
  "The number of steps between writing tensorboard summaries.",
)
flags.DEFINE_integer(
  "clgen_per_epoch_test_samples",
  16,
  "The number of samples to make at the end of each training epoch.",
)


class tfSequential(backends.BackendBase):
  """A model with an embedding layer, using a keras backend."""

  @property
  def hidden_state_size(self) -> int:
    return self.config.architecture.neurons_per_layer

  def __init__(self, *args, **kwargs):
    """Instantiate a model.

    Args:
      args: Arguments to be passed to BackendBase.__init__().
      kwargs: Arguments to be passed to BackendBase.__init__().
    """
    super(tfSequential, self).__init__(*args, **kwargs)

    local_tf.initTensorflow()

    # Attributes that will be lazily set.
    self.cell = None
    self.input_data = None
    self.targets = None
    self.lengths = None
    self.seed_length = None
    self.temperature = None
    self.initial_state = None
    self.logits = None
    self.generated = None
    self.loss = None
    self.final_state = None
    self.learning_rate = None
    self.epoch = None
    self.train_op = None
    self.data_generator = None

    self.inference_tf = None
    self.inference_sess = None
    self.inference_indices = None
    self.inference_state = None

    # Create the summary writer, shared between Train() and
    # _EndOfEpochTestSample().
    tf.compat.v1.disable_eager_execution()

    tensorboard_dir = f"{self.cache.path}/tensorboard"
    l.logger().info(
      "Using tensorboard to log training progress. View progress using:\n"
      f"    $ tensorboard --logdir='{tensorboard_dir}'",
    )
    self.summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_dir)

  def samplesWithCategorical(self):
    return True

  def InitTfGraph(
    self, sampler: typing.Optional[samplers.Sampler] = None
  ) -> "tf":
    """Instantiate a TensorFlow graph for training or inference.

    The tensorflow graph is different for training and inference, so must be
    reset when switching between modes.

    Args:
      sampler: If set, initialize the model for inference using the given
        sampler. If not set, initialize model for training.

    Returns:
      The imported TensorFlow module.
    """

    start_time = time.time()

    # Quiet tensorflow.
    # See: https://github.com/tensorflow/tensorflow/issues/1258
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Deferred importing of TensorFlow.
    tf.compat.v1.disable_eager_execution()
    from deeplearning.benchpress.models.tf_sequential import helper

    cell_type = {
      model_pb2.NetworkArchitecture.LSTM: tf.compat.v1.nn.rnn_cell.LSTMCell,
      model_pb2.NetworkArchitecture.GRU: tf.compat.v1.nn.rnn_cell.GRUCell,
      model_pb2.NetworkArchitecture.RNN: tf.compat.v1.nn.rnn_cell.BasicRNNCell,
    }.get(self.config.architecture.neuron_type, None)
    if cell_type is None:
      raise NotImplementedError

    # Reset the graph when switching between training and inference.
    tf.compat.v1.reset_default_graph()

    if sampler:
      sequence_length = sampler.sequence_length
      batch_size = sampler.batch_size
    else:
      sequence_length = self.config.training.sequence_length
      batch_size = self.config.training.batch_size
    vocab_size = self.tokenizer.vocab_size

    cells_lst = []
    for _ in range(self.config.architecture.num_layers):
      cells_lst.append(cell_type(self.config.architecture.neurons_per_layer))
    self.cell = cell = tf.keras.layers.StackedRNNCells(cells_lst)

    self.input_data = tf.compat.v1.placeholder(
      tf.int32, [batch_size, sequence_length]
    )
    self.targets = tf.compat.v1.placeholder(
      tf.int32, [batch_size, sequence_length]
    )
    self.initial_state = self.cell.get_initial_state(batch_size = batch_size, dtype = tf.float32)
    self.temperature = tf.Variable(1.0, trainable=False)
    self.seed_length = tf.compat.v1.placeholder(name = "seed_length", dtype = tf.int32, shape = ())

    if sampler:
      self.lengths = tf.compat.v1.placeholder(tf.int32, [batch_size])
    else:
      self.lengths = tf.fill([batch_size], sequence_length)

    scope_name = "rnnlm"
    with tf.compat.v1.variable_scope(scope_name):
      with tf.device("/cpu:0"):
        embedding = tf.compat.v1.get_variable(
          "embedding", [vocab_size, self.config.architecture.neurons_per_layer]
        )
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    if sampler:
      decode_helper = helper.CustomInferenceHelper(
        self.seed_length, embedding, self.temperature
      )
    else:
      decode_helper = tfa.seq2seq.sampler.TrainingSampler(time_major=False)

    decoder = tfa.seq2seq.BasicDecoder(
      cell,
      decode_helper,
      tf.compat.v1.layers.Dense(vocab_size),
      dtype = tf.float32,
    )
    outputs, self.final_state, _ = tfa.seq2seq.dynamic_decode(
      decoder,
      decoder_init_input = inputs,
      decoder_init_kwargs = {
                              'initial_state': self.initial_state,
                              'sequence_length': self.lengths,
                            },
      output_time_major=False,
      impute_finished=True,
      swap_memory=True,
      scope=scope_name,
    )

    self.generated = outputs.sample_id
    self.logits = outputs.rnn_output

    sequence_weigths = tf.ones([batch_size, sequence_length])
    self.loss = tfa.seq2seq.sequence_loss(
      self.logits, self.targets, sequence_weigths
    )

    self.learning_rate = tf.Variable(0.0, trainable=False)
    self.epoch = tf.Variable(0, trainable=False)
    trainable_variables = tf.compat.v1.trainable_variables()

    # TODO(cec): Support non-adam optimizers.
    grads, _ = tf.clip_by_global_norm(
      tf.gradients(self.loss, trainable_variables, aggregation_method=2),
      self.config.training.adam_optimizer.normalized_gradient_clip_micros / 1e6,
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    if not sampler:
      # Create tensorboard summary writers for training progress.
      tf.compat.v1.summary.scalar("loss", self.loss)
      tf.compat.v1.summary.scalar("learning_rate", self.learning_rate)
      tf.compat.v1.summary.scalar("epoch_num", self.epoch)

    num_trainable_params = int(
      np.sum([np.prod(v.shape) for v in tf.compat.v1.trainable_variables()])
    )
    l.logger().info(
      "Instantiated TensorFlow graph with {} trainable parameters " "in {} ms."
        .format(
          humanize.intcomma(num_trainable_params),
          humanize.intcomma(int((time.time() - start_time) * 1000)),
          )
    )

    return tf

  def GetShortSummary(self) -> str:
    return (
      f"{self.config.architecture.neurons_per_layer}×"
      f"{self.config.architecture.num_layers} "
      f"{model_pb2.NetworkArchitecture.NeuronType.Name(self.config.architecture.neuron_type)} "
      "network"
    )

  @property
  def epoch_checkpoints(self) -> typing.Set[int]:
    """Get the set of epoch numbers which we have trained models for.

    Note that Tensorflow checkpoint paths don't translate to actual files, but
    rather a pair of <.index,.meta> files.

    Returns:
      A mapping of epoch numbers to paths.
    """
    if not (self.cache.path / "checkpoints" / "checkpoints"):
      # No saver file means no checkpoints.
      return {}

    # Count the number of checkpoint files which TensorFlow has created.
    checkpoint_files = [
      f.stem
      for f in (self.cache.path / "checkpoints").iterdir()
      if f.name.startswith("checkpoint-") and f.name.endswith(".meta")
    ]
    # The checkpoint paths are appended with the epoch number.
    epoch_nums = [int(x.split("-")[-1]) for x in checkpoint_files]
    return set(epoch_nums)

  def GetParamsPath(
    self, checkpoint_state
  ) -> typing.Tuple[typing.Optional[str], typing.List[str]]:
    """Return path to checkpoint closest to target num of epochs."""
    # Checkpoints are saved with relative path, so we must prepend cache paths.
    paths = [
      str(self.cache.path / "checkpoints" / p)
      for p in checkpoint_state.all_model_checkpoint_paths
    ]
    # The checkpoint paths are appended with the epoch number.
    epoch_nums = [int(x.split("-")[-1]) for x in paths]
    diffs = [self.config.training.num_epochs - e for e in epoch_nums]
    pairs = zip(paths, diffs)
    positive_only = [p for p in pairs if p[1] >= 0]
    return min(positive_only, key=lambda x: x[1])[0], paths

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.

    Returns:
      A list of absolute paths.
    """
    # The TensorFlow save file.
    paths = [
      self.cache.path / "checkpoints" / "checkpoint",
    ]
    # Export only the TensorFlow checkpoint files for the target number of
    # epochs.
    paths += [
      path.absolute()
      for path in (self.cache.path / "checkpoints").iterdir()
      if path.name.startswith(f"checkpoint-{self.config.training.num_epochs}")
    ]
    # Include the epoch telemetry. This is not strictly required, but the files
    # are small and contain useful information for describing the model, such as
    # the total training time and model loss.
    paths += [
      path.absolute()
      for path in (self.cache.path / "logs").iterdir()
      if (
        path.name.startswith("epoch_")
        and path.name.endswith("_telemetry.pbtxt")
      )
    ]
    return sorted(paths)

  def Train(
    self,
    corpus,
    test_sampler: typing.Optional[samplers.Sampler] = None,
    **unused_kwargs,
  ) -> None:
    """Locked training.

    If there are cached epoch checkpoints, the one closest to the target number
    of epochs will be loaded, and the model will be trained for only the
    remaining number of epochs, if any. This means that calling this function
    twice will only actually train the model the first time, and all subsequent
    calls will be no-ops.

    This method must only be called when the model is locked.
    """
    del unused_kwargs

    self.num_epochs = self.config.training.num_epochs
    self.telemetry = telemetry.TrainingLogger(self.cache.path / "logs")

    if self.is_trained:
      return

    if self.data_generator is None:
      self.data_generator = TensorflowBatchGenerator(
        corpus, self.config.training
      )
    tf = self.InitTfGraph()

    # Create and merge the tensorboard summary ops.
    merged = tf.compat.v1.summary.merge_all()

    # training options
    # TODO(cec): Enable support for multiple optimizers:
    initial_learning_rate = (
      self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    )
    decay_rate = (
      self.config.training.adam_optimizer.learning_rate_decay_per_epoch_micros
      / 1e6
    )

    # # resume from prior checkpoint
    ckpt_path, ckpt_paths = None, None
    if (self.cache.path / "checkpoints" / "checkpoint").exists():
      checkpoint_state = tf.train.get_checkpoint_state(
        self.cache.path / "checkpoints",
      )
      assert checkpoint_state
      assert checkpoint_state.model_checkpoint_path
      ckpt_path, ckpt_paths = self.GetParamsPath(checkpoint_state)

    with tf.compat.v1.Session() as sess:
      tf.compat.v1.global_variables_initializer().run()

      # Keep all checkpoints.
      saver = tf.compat.v1.train.Saver(
        tf.compat.v1.global_variables(), max_to_keep=100, save_relative_paths=True
      )

      # restore model from closest checkpoint.
      if ckpt_path:
        l.logger().info("Restoring checkpoint {}".format(ckpt_path))
        saver.restore(sess, ckpt_path)

      # make sure we don't lose track of other checkpoints
      if ckpt_paths:
        saver.recover_last_checkpoints(ckpt_paths)

      # Offset epoch counts by 1 so that they are in the range [1..n]
      current_epoch = sess.run(self.epoch) + 1
      max_epoch = self.config.training.num_epochs + 1

      # Per-epoch training loop.
      for epoch_num in range(current_epoch, max_epoch):
        self.telemetry.EpochBeginCallback()

        # decay and set learning rate
        new_learning_rate = initial_learning_rate * (
          (float(100 - decay_rate) / 100.0) ** (epoch_num - 1)
        )
        sess.run(tf.compat.v1.assign(self.learning_rate, new_learning_rate))
        sess.run(tf.compat.v1.assign(self.epoch, epoch_num))

        # TODO(cec): refactor data generator to a Python generator.
        self.data_generator.CreateBatches()
        l.logger().info("Epoch {}/{}:".format(epoch_num, self.config.training.num_epochs))
        state = sess.run(self.initial_state)
        # Per-batch inner loop.
        bar = progressbar.ProgressBar(max_value=self.data_generator.num_batches)
        last_log_time = time.time()
        for i in bar(range(self.data_generator.num_batches)):
          x, y = self.data_generator.NextBatch()
          feed = {self.input_data: x, self.targets: y}
          for j, (c, h) in enumerate(self.initial_state):
            feed[c], feed[h] = state[j].c, state[j].h
          summary, loss, state, _ = sess.run(
            [merged, self.loss, self.final_state, self.train_op], feed
          )

          # Periodically write progress to tensorboard.
          if i % FLAGS.clgen_tf_backend_tensorboard_summary_step_count == 0:
            step = (epoch_num - 1) * self.data_generator.num_batches + i
            self.summary_writer.add_summary(summary, step)

        # Log the loss and delta.
        l.logger().info("Loss: {:.6f}.".format(loss))

        # Save after every epoch.
        start_time = time.time()
        global_step = epoch_num
        checkpoint_prefix = self.cache.path / "checkpoints" / "checkpoint"
        checkpoint_path = saver.save(
          sess, str(checkpoint_prefix), global_step=global_step
        )
        l.logger().info(
          "Saved checkpoint {} in {} ms."
            .format(
              checkpoint_path,
              humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
        )
        assert pathlib.Path(
          f"{checkpoint_prefix}-{global_step}.index"
        ).is_file()
        assert pathlib.Path(f"{checkpoint_prefix}-{global_step}.meta").is_file()

        self.telemetry.EpochEndCallback(epoch_num, loss)
        # If we have a sampler that we can use at the end of epochs, then
        # break now to run the test sampler.
        # This is confusing logic! Consider a refactor to simplify things.
        if test_sampler:
          break
      else:
        return

    if test_sampler and FLAGS.clgen_per_epoch_test_samples > 0:
      self._EndOfEpochTestSample(corpus, test_sampler, step, epoch_num)
      self.Train(corpus, test_sampler=test_sampler)

  def _EndOfEpochTestSample(
    self, corpus, sampler: samplers.Sampler, step: int, epoch_num: int
  ):
    """Run sampler"""
    tf.compat.v1.disable_eager_execution()

    tokenizer = corpus.tokenizer
    sampler.Specialize(tokenizer)
    sampler.batch_size = 1
    seed = 0

    self.InitSampling(sampler, seed)
    self.InitSampleBatch(sampler)

    samples, stats = [], []
    for i in range(FLAGS.clgen_per_epoch_test_samples):
      done = np.zeros(1, dtype=np.bool)
      start_time = time.time()
      sample_in_progress = sampler.tokenized_start_text.copy()

      while not done[0]:
        indices, _ = self.SampleNextIndices(sampler, done)
        # Iterate over all samples in batch to determine whether they're
        # done.
        for index in indices[0]:
          sample_in_progress.append(tokenizer.decoder[index])
          if sampler.SampleIsComplete(sample_in_progress):
            stats.append(
              (len(sample_in_progress), int((time.time() - start_time) * 1000))
            )
            sample = "".join(sample_in_progress)
            print(f"=== CLGEN SAMPLE ===\n\n{sample}\n")
            samples.append(sample)
            done[0] = True
            break
    samples_as_markdown = [
      self.FormatCodeAsMarkdown(sample) for sample in samples
    ]
    samples_tensor = tf.convert_to_tensor(samples_as_markdown, dtype=tf.string)
    summary_op = tf.compat.v1.summary.text("samples", samples_tensor)
    summary = self.inference_sess.run(summary_op)
    self.summary_writer.add_summary(summary, step)

  @staticmethod
  def FormatCodeAsMarkdown(text: str) -> str:
    return f"<pre>{text.strip()}</pre>"

  def InitSampling(self,
                   sampler: samplers.Sampler,
                   seed: typing.Optional[int] = None,
                   *unused_args,
                   **unused_kwargs,
                   ) -> None:
    """Initialize model for sampling."""
    del unused_args
    del unused_kwargs
    tf.compat.v1.disable_eager_execution()
    
    # Delete any previous sampling session.
    if self.inference_tf:
      del self.inference_tf
    if self.inference_sess:
      del self.inference_sess

    self.inference_tf = self.InitTfGraph(sampler=sampler)
    self.inference_sess = self.inference_tf.compat.v1.Session()

    # Seed the RNG.
    if seed is not None:
      np.random.seed(seed)
      self.inference_tf.compat.v1.set_random_seed(seed)

    # If --clgen_tf_backend_reset_inference_state_between_batches, the state
    # is reset at the beginning of every sample batch. Else, this is the only
    # place it is initialized.
    self.inference_state = self.inference_sess.run(
      self.cell.get_initial_state(batch_size = sampler.batch_size, dtype = self.inference_tf.float32)
    )

    self.inference_tf.compat.v1.global_variables_initializer().run(
      session=self.inference_sess
    )
    # Restore trained model weights.
    saver = self.inference_tf.compat.v1.train.Saver(
      self.inference_tf.compat.v1.global_variables()
    )
    checkpoint_state = self.inference_tf.train.get_checkpoint_state(
      self.cache.path / "checkpoints",
    )

    # These assertions will fail if the model has no checkpoints. Since this
    # should only ever be called after Train(), there is no good reason for
    # these assertions to fail.
    assert checkpoint_state
    assert checkpoint_state.model_checkpoint_path

    if FLAGS.select_checkpoint_step == -1:
      saver.restore(self.inference_sess, checkpoint_state.model_checkpoint_path)
    else:
      saver.restore(self.inference_sess, str(self.cache.path / "checkpoints" / "checkpoint-{}".format(FLAGS.select_checkpoint_step)))
    self.inference_sess.run(
      tf.compat.v1.assign(self.temperature, sampler.temperature)
    )

  def InitSampleBatch(self, sampler: samplers.Sampler) -> None:
    if FLAGS.clgen_tf_backend_reset_inference_state_between_batches:
      self.inference_state = self.inference_sess.run(
        self.cell.get_initial_state(batch_size = sampler.batch_size, dtype = self.inference_tf.float32)
      )
    self.inference_indices = np.tile(
      sampler.encoded_start_text, [sampler.batch_size, 1]
    )

  def SampleNextIndices(self, sampler: samplers.Sampler, done: np.ndarray):
    length = self.inference_indices.shape[1]
    assert length < sampler.sequence_length
    expanded_indices = np.zeros((sampler.batch_size, sampler.sequence_length))
    expanded_indices[:, :length] = self.inference_indices
    synthesized_lengths = np.full([sampler.batch_size], sampler.sequence_length)
    synthesized_lengths[done] = 0
    feed = {
      self.initial_state: self.inference_state,
      self.input_data: expanded_indices,
      self.lengths: synthesized_lengths,
      self.seed_length: length,
    }

    generated, self.inference_state = self.inference_sess.run(
      [self.generated, self.final_state], feed
    )

    self.inference_indices = generated[:, -1].reshape((sampler.batch_size, 1))
    if length > 1:
      generated = generated[:, length - 1 :]
    return generated, generated

  def RandomizeSampleState(self) -> None:
    tf.compat.v1.disable_eager_execution()

    self.inference_state = [
      tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
        st1 + np.random.normal(scale=0.2, size=np.shape(st1)),
        st2 + np.random.normal(scale=0.2, size=np.shape(st2)),
      )
      for st1, st2 in self.inference_state
    ]

  def ResetSampleState(self, sampler: samplers.Sampler, state, seed) -> None:
    self.inference_state = copy.deepcopy(state)
    self.inference_indices = np.tile(seed, [sampler.batch_size, 1])

  def EvaluateSampleState(self, sampler: samplers.Sampler):
    length = self.inference_indices.shape[1] - 1
    if length == 0:
      return
    last_indices = self.inference_indices[:, -1:]
    self.inference_indices = self.inference_indices[:, :-1]

    expanded_indices = np.zeros((sampler.batch_size, sampler.sequence_length))
    expanded_indices[:, :length] = self.inference_indices
    synthesized_lengths = np.full([sampler.batch_size], length)

    feed = {
      self.initial_state: self.inference_state,
      self.input_data: expanded_indices,
      self.lengths: synthesized_lengths,
      self.seed_length: length,
    }

    self.inference_state = self.inference_sess.run([self.final_state], feed)
    self.inference_indices = last_indices

    state_copy = copy.deepcopy(self.inference_state)
    input_carry_copy = self.inference_indices[0]
    return state_copy, input_carry_copy

  @property
  def is_trained(self) -> bool:
    """Determine if model has been trained."""
    # Count the number of checkpoint files which TensorFlow has created.
    checkpoint_files = [
      f.stem
      for f in (self.cache.path / "checkpoints").iterdir()
      if f.name.startswith("checkpoint-") and f.name.endswith(".meta")
    ]
    epoch_nums = [int(x.split("-")[-1]) for x in checkpoint_files]
    return self.config.training.num_epochs in epoch_nums
