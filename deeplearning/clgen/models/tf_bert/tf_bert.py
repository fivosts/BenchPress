# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import typing
import pathlib

from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.tf import tf
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.models.tf_bert import model
from deeplearning.clgen.models.tf_bert import optimizer
from deeplearning.clgen.models.tf_bert import hooks

from eupy.native import logger as l
from labm8.py import app
from labm8.py import gpu_scheduler

FLAGS = app.FLAGS

app.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

app.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

app.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

app.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

app.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

app.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

app.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class tfBert(backends.BackendBase):

  def __init__(self, *args, **kwargs):

    super(tfBert, self).__init__(*args, **kwargs)
    self.bertConfig = {
          "vocab_size"                      : None,
          "hidden_size"                     : None,
          "num_hidden_layers"               : None,
          "num_attention_heads"             : None,
          "intermediate_size"               : None,
          "hidden_act"                      : None,
          "hidden_dropout_prob"             : None,
          "attention_probs_dropout_prob"    : None,
          "max_position_embeddings"         : None,
          "type_vocab_size"                 : None,
          "initializer_range"               : None,
    }

    self.max_seq_length                     = None
    self.train_batch_size                   = None
    self.eval_batch_size                    = None
    self.max_predictions_per_seq            = None
    self.learning_rate                      = None
    self.num_train_steps                    = None
    self.num_warmup_steps                   = None
    self.is_trained                         = False

    return

  def _ConfigModelParams(self):

    self.bertConfig = {
          "vocab_size"                      : self.atomizer.vocab_size,
          "hidden_size"                     : self.config.architecture.hidden_size,
          "num_hidden_layers"               : self.config.architecture.num_hidden_layers,
          "num_attention_heads"             : self.config.architecture.num_attention_heads,
          "intermediate_size"               : self.config.architecture.intermediate_size,
          "hidden_act"                      : self.config.architecture.hidden_act,
          "hidden_dropout_prob"             : 1.0 - self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob"    : 1.0 - self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"         : self.config.architecture.max_position_embeddings,
          "type_vocab_size"                 : self.config.architecture.type_vocab_size,
          "initializer_range"               : self.config.architecture.initializer_range,
    }

    self.max_seq_length                     = self.config.training.sequence_length
    self.train_batch_size                   = self.config.training.batch_size
    self.eval_batch_size                    = self.config.training.batch_size
    self.max_predictions_per_seq            = self.config.training.max_predictions_per_seq
    self.learning_rate                      = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_train_steps                    = self.config.training.num_train_steps
    self.num_warmup_steps                   = self.config.training.num_warmup_steps
    self.ckpt_path                          = self.cache.path / "checkpoints"
    self.logfile_path                       = self.cache.path / "logs"
    self.sample_path                        = self.cache.path / "samples"
    self.telemetry                          = telemetry.TrainingLogger(str(self.logfile_path))

    self.num_steps_per_epoch                = self.data_generator.num_batches
    self.num_epochs                         = int(self.num_train_steps / self.num_steps_per_epoch)

    self.validation_results_file            = "val_results.txt"
    self.validation_results_path            = os.path.join(str(self.logfile_path), self.validation_results_file)

    return

  def Train(self, corpus, **unused_kwargs) -> None:

    del unused_kwargs

    ## TODO, also search the checkpoints to determine if is_trained
    ## Additionally, you will have to exclude num_train_steps from model hash
    if not self.is_trained:

      ## Acquire GPU Lock before anything else is done
      gpu_scheduler.LockExclusiveProcessGpuAccess()

      ## Initialize params and data generator
      self.data_generator = data_generators.MaskLMBatchGenerator.TrainMaskLMBatchGenerator(
                                corpus, self.config.training, self.cache.path
                             )
      self._ConfigModelParams()

      ## Generate BERT Model from dict params
      bert_config = model.BertConfig.from_dict(self.bertConfig)

      l.getLogger().info("Checkpoint path: \n{}".format(self.ckpt_path))
      l.getLogger().info("Logging path: \n{}".format(self.logfile_path))
      
      tpu_cluster_resolver = None
      if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

      is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
      run_config = tf.compat.v1.estimator.tpu.RunConfig(
          cluster = tpu_cluster_resolver,
          master = FLAGS.master,
          model_dir = str(self.ckpt_path),
          save_checkpoints_steps = self.num_steps_per_epoch,
          save_summary_steps = self.num_steps_per_epoch,
          keep_checkpoint_max = 0,
          log_step_count_steps = self.num_steps_per_epoch,
          tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
              iterations_per_loop = self.num_steps_per_epoch,
              num_shards = FLAGS.num_tpu_cores,
              per_host_input_for_training = is_per_host))

      model_fn = self._model_fn_builder(
          bert_config = bert_config,
          init_checkpoint = self.ckpt_path,
          learning_rate = self.learning_rate,
          num_train_steps = self.num_train_steps,
          num_warmup_steps = self.num_warmup_steps,
          use_tpu = FLAGS.use_tpu,
          use_one_hot_embeddings = FLAGS.use_tpu)

      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
          use_tpu = FLAGS.use_tpu,
          model_fn = model_fn,
          config = run_config,
          train_batch_size = self.train_batch_size,
          eval_batch_size = self.eval_batch_size)

      l.getLogger().info("BERT Training initialization")
      ## TODO print short summary of model and/or dataset features

      train_input_fn = self.data_generator.generateTfDataset(
          max_seq_length = self.max_seq_length,
          num_cpu_threads = 8,
          is_training = True)

      l.getLogger().info("Running model for {} steps".format(self.num_train_steps))
      l.getLogger().info("Splitting {} steps into {} equivalent epochs, {} steps each".format(
                                        self.num_train_steps, self.num_epochs, self.num_steps_per_epoch
                                        )
                        )

      estimator.train(input_fn=train_input_fn, max_steps = self.num_train_steps)
      self.is_trained = True
      self.telemetry.TfRecordEpochs()

    if not self.is_validated:
      l.getLogger().info("BERT Validation")
      ## TODO print short summary of model and/or dataset features

      eval_input_fn = self.data_generator.generateTfDataset(
          max_seq_length=self.max_seq_length,
          num_cpu_threads = 8,
          is_training=False)

      result = estimator.evaluate(
          input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

      self._writeValidation(result)

    return

  def InitSampling(self,
                   sampler: samplers.Sampler, 
                   seed: typing.Optional[int] = None
                   ) -> None:

    l.getLogger().warning("Init Sampling: Called once, sets model")

    self.data_generator = data_generator.MaskLMBatchGenerator.SampleMaskLMBatchGenerator(sampler, seed)

    if self.bertConfig is None:
      self._ConfigModelParams()
    bert_config = model.BertConfig.from_dict(self.bertConfig)

    if self.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (self.max_seq_length, bert_config.max_position_embeddings))

    # processor = processors[task_name]()

    # label_list = processor.get_labels()

    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.compat.v1.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster = tpu_cluster_resolver,
        master = FLAGS.master,
        model_dir = str(self.ckpt_path),
        # save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
            # iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host))

    ## TODO integrate this fn_builder to the current one
    # model_fn = model_fn_builder(
    #     bert_config = bert_config,
    #     num_labels = len(label_list),
    #     init_checkpoint = self.ckpt_path,
    #     # learning_rate = self.learning_rate,
    #     # num_train_steps = num_train_steps,
    #     # num_warmup_steps = num_warmup_steps,
    #     use_tpu = FLAGS.use_tpu,
    #     use_one_hot_embeddings = FLAGS.use_tpu)

    model_fn = self._model_fn_builder(
        bert_config = bert_config,
        init_checkpoint = self.ckpt_path,
        # learning_rate = self.learning_rate,
        # num_train_steps = self.num_train_steps,
        # num_warmup_steps = self.num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu,
        sampling_mode = True)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    # estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
    #     use_tpu = FLAGS.use_tpu,
    #     model_fn = model_fn,
    #     config = run_config,
    #     # train_batch_size = self.train_batch_size,
    #     # eval_batch_size = self.eval_batch_size,
    #     predict_batch_size = sampler.batch_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        # train_batch_size = self.train_batch_size,
        predict_batch_size = sampler.batch_size)

    ## TODO save that stuff somewhere
    l.getLogger().info("Initialized BERT sampler in {}".format(self.sample_path))

    return 

  def InitSampleBatch(self,
                   sampler: samplers.Sampler, 
                   ) -> None:

    l.getLogger().warning("Called while batches are not done. Sets up batch")
    return 

  def SampleNextIndices(self, sampler: samplers.Sampler, done):
    l.getLogger().warning("Within a batch, called for each i/o step")

    if FLAGS.do_predict:
      # predict_examples = processor.get_test_examples(FLAGS.data_dir)
      # num_actual_predict_examples = len(predict_examples)
      if FLAGS.use_tpu:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
          predict_examples.append(PaddingInputExample())

      predict_file = str(self.sample_path / "predict.tf_record")
      file_based_convert_examples_to_features(predict_examples, label_list,
                                              self.max_seq_length, tokenizer,
                                              predict_file)

      tf.logging.info("***** Running prediction*****")
      tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                      len(predict_examples), num_actual_predict_examples,
                      len(predict_examples) - num_actual_predict_examples)
      tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

      predict_drop_remainder = True if FLAGS.use_tpu else False

      ## TODO this function is going to the data_generator
      ## and will be migrated from file based, to sampler text based builder
      predict_input_fn = file_based_input_fn_builder(
          input_file = predict_file,
          seq_length = self.max_seq_length,
          is_training = False,
          drop_remainder = predict_drop_remainder)

      ## Batch size could determine the number of tf.data entries provided by
      ## the input_fn builder
      result = estimator.predict(input_fn=predict_input_fn)

      output_predict_file = str(self.sample_path / "test_results.tsv")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
              str(class_probability)
              for class_probability in probabilities) + "\n"
          writer.write(output_line)
          num_written_lines += 1
      assert num_written_lines == num_actual_predict_examples

    return []


  def GetShortSummary(self) -> str:
    l.getLogger().debug("deeplearning.clgen.models.tf_bert.tfBert.GetShortSummary()")
    return (
      f"h_s: {self.bertConfig['hidden_size']}, "
      f"#h_l: {self.bertConfig['num_hidden_layers']}, "
      f"#att_h: {self.bertConfig['num_attention_heads']}, "
      f"imd_s: {self.bertConfig['intermediate_size']}, "
      f"h_act: {self.bertConfig['hidden_act']}, "
      f"{model_pb2.NetworkArchitecture.NeuronType.Name(self.config.architecture.backend)} "
      "network"
    )

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.
    Returns:
      A list of absolute paths.
    """
    # The TensorFlow save file.
    l.getLogger().debug("deeplearning.clgen.models.tf_bert.tfBert.InferenceManifest()")
    paths = [ path.absolute() for path in (self.cache.path / "checkpoints").iterdir() ]
    paths += [ path.absolute() for path in (self.cache.path / "logs").iterdir() ]
    paths += [ self.data_generator.tfRecord ]
    return sorted(paths)

  def _model_fn_builder(self,
                      bert_config, 
                      init_checkpoint, 
                      learning_rate,
                      num_train_steps,
                      num_warmup_steps,
                      use_tpu,
                      use_one_hot_embeddings,
                      sampling_mode = False):
    """Returns `model_fn` closure for TPUEstimator."""

    def _model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      # l.getLogger().info("*** Features ***")
      # for name in sorted(features.keys()):
      #   l.getLogger().info("  name = %s, shape = %s" % (name, features[name].shape))

      input_ids = features["input_ids"]
      # input_mask = features["input_mask"]
      # segment_ids = features["segment_ids"]
      masked_lm_positions = features["masked_lm_positions"]
      masked_lm_ids = features["masked_lm_ids"]
      masked_lm_weights = features["masked_lm_weights"]
      next_sentence_labels = features["next_sentence_labels"]

      is_training = (mode == tf.compat.v1.estimator.ModeKeys.TRAIN)

      bert_model = model.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          # input_mask=input_mask, # You CAN ignore. Used for padding. 0s after real sequence. Now all 1s.
          # token_type_ids=segment_ids, # You can ignore. Used for double sentences (sA -> 0, sB ->1). Now all will be zero
          use_one_hot_embeddings=use_one_hot_embeddings)

      (masked_lm_loss,
       masked_lm_example_loss, masked_lm_log_probs) = self._get_masked_lm_output(
           bert_config, bert_model.get_sequence_output(), bert_model.get_embedding_table(),
           masked_lm_positions, masked_lm_ids, masked_lm_weights)

      (next_sentence_loss, next_sentence_example_loss,
       next_sentence_log_probs) = self._get_next_sentence_output(
           bert_config, bert_model.get_pooled_output(), next_sentence_labels)

      total_loss = masked_lm_loss + next_sentence_loss
      tvars = tf.compat.v1.trainable_variables()

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def _tpu_scaffold():
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = _tpu_scaffold
        else:
          l.getLogger().info("Loading model checkpoint from: {}".format(init_checkpoint))
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

      # l.getLogger().info("**** Trainable Variables ****")
      # for var in tvars:
      #   init_string = ""
      #   if var.name in initialized_variable_names:
      #     init_string = ", *INIT_FROM_CKPT*"
      #   l.getLogger().info("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))

      output_spec = None
      if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
        train_op = optimizer.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        training_hooks = self._GetSummaryHooks(save_steps = self.num_steps_per_epoch,
                                              output_dir = str(self.logfile_path),
                                              masked_lm_loss = masked_lm_loss,
                                              next_sentence_loss = next_sentence_loss,
                                              total_loss = total_loss,
                                              learning_rate = learning_rate
                                              )

        training_hooks += self._GetProgressBarHooks(max_length = self.num_train_steps,
                                                    tensors = {'Loss': total_loss},
                                                    log_steps = self.num_steps_per_epoch,
                                                    at_end = True,
                                                    )

        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode = mode,
            loss = total_loss,
            train_op = train_op,
            training_hooks = training_hooks,
            scaffold_fn = scaffold_fn)
      elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:

        def _metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                      masked_lm_weights, next_sentence_example_loss,
                      next_sentence_log_probs, next_sentence_labels):
          """Computes the loss and accuracy of the model."""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions, ## TODO!! This is your predictions for mask
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.compat.v1.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          next_sentence_log_probs = tf.reshape(
              next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
          next_sentence_predictions = tf.argmax(
              next_sentence_log_probs, axis=-1, output_type=tf.int32)
          next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
          next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
              labels=next_sentence_labels, predictions=next_sentence_predictions)
          next_sentence_mean_loss = tf.compat.v1.metrics.mean(
              values=next_sentence_example_loss)

          return {
              "masked_lm_accuracy": masked_lm_accuracy,
              "masked_lm_loss": masked_lm_mean_loss,
              "next_sentence_accuracy": next_sentence_accuracy,
              "next_sentence_loss": next_sentence_mean_loss,
          }

        eval_metrics = (_metric_fn, [
            masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
            masked_lm_weights, next_sentence_example_loss,
            next_sentence_log_probs, next_sentence_labels
        ])

        evaluation_hooks = self._GetProgressBarHooks(
          max_length = FLAGS.max_eval_steps, is_training = False
        )
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            evaluation_hooks = evaluation_hooks,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        def _metric_fn(masked_lm_log_probs, masked_lm_ids, next_sentence_log_probs):
          """Computes the model's mask and next sentence predictions"""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])

          next_sentence_log_probs = tf.reshape(
              next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
          next_sentence_predictions = tf.argmax(
              next_sentence_log_probs, axis=-1, output_type=tf.int32)

          return {
              "masked_lm_predictions": masked_lm_predictions,
              "next_sentence_predictions": next_sentence_predictions,
          }

        eval_metrics = (_metric_fn, [masked_lm_log_probs, masked_lm_ids, next_sentence_log_probs])

        evaluation_hooks = self._GetProgressBarHooks(
          max_length = FLAGS.max_eval_steps, is_training = False
        )
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            evaluation_hooks = evaluation_hooks,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)

      return output_spec

    return _model_fn

  def _get_masked_lm_output(self, 
                           bert_config,
                           input_tensor, 
                           output_weights, 
                           positions, 
                           label_ids,
                           label_weights
                           ):
    """Get loss and log probs for the masked LM."""
    input_tensor = self._gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
      # We apply one more non-linear transformation before the output layer.
      # This matrix is not used after pre-training.
      with tf.compat.v1.variable_scope("transform"):
        input_tensor = tf.compat.v1.layers.dense(
            input_tensor,
            units=bert_config.hidden_size,
            activation=model.get_activation(bert_config.hidden_act),
            kernel_initializer=model.create_initializer(
                bert_config.initializer_range))
        input_tensor = model.layer_norm(input_tensor)

      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = tf.compat.v1.get_variable(
          "output_bias",
          shape=[bert_config.vocab_size],
          initializer=tf.zeros_initializer())
      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      label_ids = tf.reshape(label_ids, [-1])
      label_weights = tf.reshape(label_weights, [-1])

      one_hot_labels = tf.one_hot(
          label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
      numerator = tf.reduce_sum(label_weights * per_example_loss)
      denominator = tf.reduce_sum(label_weights) + 1e-5
      loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


  def _get_next_sentence_output(self, 
                               bert_config,
                               input_tensor,
                               labels
                              ):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("cls/seq_relationship"):
      output_weights = tf.compat.v1.get_variable(
          "output_weights",
          shape=[2, bert_config.hidden_size],
          initializer=model.create_initializer(bert_config.initializer_range))
      output_bias = tf.compat.v1.get_variable(
          "output_bias", shape=[2], initializer=tf.zeros_initializer())

      logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      labels = tf.reshape(labels, [-1])
      one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      return (loss, per_example_loss, log_probs)


  def _gather_indexes(self, sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = model.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor

  def _GetSummaryHooks(self, 
                       save_steps: int, 
                       output_dir: str, 
                       **kwargs
                       ) -> typing.List[tf.estimator.SummarySaverHook]:
    return [tf.estimator.SummarySaverHook(save_steps = save_steps,
                                          output_dir = output_dir,
                                          summary_op = [ tf.compat.v1.summary.scalar(name, value) 
                                                          for name, value in kwargs.items()
                                                        ]
                                          )
           ]

  def _GetProgressBarHooks(self, 
                           max_length: int, 
                           tensors = None,
                           log_steps = None,
                           at_end = None,
                           is_training = True,
                           ) -> typing.List[hooks.tfProgressBar]:
    return [hooks.tfProgressBar(
                max_length = max_length,
                tensors = tensors,
                log_steps = log_steps,
                at_end = at_end,
                is_training = is_training,
              )
            ]    
  
  @property
  def is_validated(self):
    if os.path.exists(self.validation_results_path):
      return True
    return False

  def _writeValidation(self,
                       result
                       ) -> None:
    with tf.io.gfile.GFile(self.validation_results_path, "w") as writer:
      l.getLogger().info("Validation set result summary")
      for key in sorted(result.keys()):
        l.getLogger().info("{}: {}".format(key, str(result[key])))
        writer.write("{}: {}\n".format(key, str(result[key])))
    return 