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
import shutil
import glob
import typing
import pathlib
import datetime
import tensorflow_probability as tfp
import numpy as np
from absl import flags

from deeplearning.clgen import samplers
from deeplearning.clgen import sample_observers
from deeplearning.clgen import telemetry
from deeplearning.clgen.tf import tf
from deeplearning.clgen import pbutil
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.models.tf_bert import model
from deeplearning.clgen.models.tf_bert import optimizer
from deeplearning.clgen.models.tf_bert import hooks

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "select_checkpoint_step",
  -1,
  "Select step checkpoint for sample, validation or training."
  "Default: -1, flag ignored and latest checkpoint is loaded."
)

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_boolean("force_eval", False, "Run Validation no matter what.")

flags.DEFINE_integer("sample_per_epoch", 3, "Set above zero to sample model after every epoch.")

flags.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_boolean("mirror_gpus", False, "Set True to distribute training across all system's GPUs. (Only usable when use_tpu is False).")

flags.DEFINE_boolean("categorical_sampling", True, "Use categorical distribution on logits when sampling.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class tfBert(backends.BackendBase):

  class BertEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT estimator pipeline."""
    estimator      : tf.compat.v1.estimator.tpu.TPUEstimator
    data_generator : data_generators.MaskLMBatchGenerator

  def __init__(self, *args, **kwargs):

    super(tfBert, self).__init__(*args, **kwargs)
    
    self.bertAttrs                       = None
    self.bert_config                     = None

    self.train                           = None
    self.sample                          = None
    self.predict_generator               = None
    self.sampler                         = None

    self.train_batch_size                = None
    self.eval_batch_size                 = None
    self.learning_rate                   = None
    self.num_train_steps                 = None
    self.num_warmup_steps                = None
    self.telemetry                       = None

    self.ckpt_path                       = self._ConfigCheckpointParams()
    self.logfile_path                    = self.cache.path / "logs"
    self.sample_path                     = self.cache.path / "samples"

    self.is_validated                    = False
    l.getLogger().info("BERT Model config initialized in {}".format(self.cache.path))
    return

  def _ConfigCheckpointParams(self):
    if FLAGS.select_checkpoint_step >= 0:

      ckpt_current = self.cache.path / "checkpoints"
      if not (ckpt_current / "model.ckpt-{}.index".format(FLAGS.select_checkpoint_step)).exists():
        raise FileNotFoundError(ckpt_current / "model.ckpt-{}.index".format(FLAGS.select_checkpoint_step))

      workspace_rel_path = self.cache.path.relative_to(pathlib.Path(os.environ.get("CLGEN_CACHE")).parent)
      ckpt_path = pathlib.Path("/tmp" / workspace_rel_path / "checkpoints")
      ckpt_path.mkdir(exist_ok = True, parents = True)

      shutil.copy2(ckpt_current / "checkpoint" , ckpt_path)
      shutil.copy2(ckpt_current / "graph.pbtxt", ckpt_path)
      for ckpt_file in glob.glob(str(ckpt_current / "model.ckpt-{}.*".format(FLAGS.select_checkpoint_step))):
        shutil.copy2(ckpt_file, ckpt_path)
      l.getLogger().warn("Explicit checkpoint selected. Explicit checkpoints can only be used for validation or sampling.")
    elif FLAGS.select_checkpoint_step == -1:
      ckpt_path = self.cache.path / "checkpoints"
    else:
      raise ValueError("Invalid value {} for --select_checkpoint_step".format(FLAGS.select_checkpoint_step))
    l.getLogger().info("Configured model checkpoints in {}".format(ckpt_path))
    return ckpt_path

  def _ConfigModelParams(self):
    self.bertAttrs = {
          "vocab_size"                   : self.atomizer.vocab_size,
          "hidden_size"                  : self.config.architecture.hidden_size,
          "num_hidden_layers"            : self.config.architecture.num_hidden_layers,
          "num_attention_heads"          : self.config.architecture.num_attention_heads,
          "intermediate_size"            : self.config.architecture.intermediate_size,
          "hidden_act"                   : self.config.architecture.hidden_act,
          "hidden_dropout_prob"          : 1.0 - self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob" : 1.0 - self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"      : self.config.architecture.max_position_embeddings,
          "type_vocab_size"              : self.config.architecture.type_vocab_size,
          "initializer_range"            : self.config.architecture.initializer_range,
    }
    self.bert_config                     = model.BertConfig.from_dict(self.bertAttrs)
    return

  def _ConfigTrainParams(self, 
                         data_generator: data_generators.MaskLMBatchGenerator
                        ) -> None:
    """
    Model parameter initialization for training and validation.
    """
    if self.bert_config is None:
      self._ConfigModelParams()

    self.train_batch_size                 = self.config.training.batch_size
    self.eval_batch_size                  = self.config.training.batch_size
    self.learning_rate                    = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_warmup_steps                 = self.config.training.num_warmup_steps

    self.telemetry                        = telemetry.TrainingLogger(self.logfile_path)
    self.steps_per_epoch                  = data_generator.steps_per_epoch
    self.num_epochs                       = data_generator.num_epochs
    self.num_train_steps                  = self.steps_per_epoch * self.num_epochs
    self.max_eval_steps                   = FLAGS.max_eval_steps

    self.validation_results_file          = "val_results.txt"
    self.validation_results_path          = os.path.join(str(self.logfile_path), self.validation_results_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project)

    train_distribute = tf.distribute.MirroredStrategy(num_gpus = gpu.numGPUs()) if FLAGS.use_tpu and FLAGS.mirror_gpus else None

    is_per_host      = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config  = tf.compat.v1.estimator.tpu.RunConfig(
                    cluster   = tpu_cluster_resolver,
                    master    = FLAGS.master,
                    model_dir = str(self.ckpt_path),
                    save_checkpoints_steps  = self.steps_per_epoch,
                    save_summary_steps      = self.steps_per_epoch,
                    keep_checkpoint_max     = 0,
                    log_step_count_steps    = self.steps_per_epoch,
                    train_distribute        = train_distribute,
                    tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
                        iterations_per_loop = self.steps_per_epoch,
                        num_shards          = FLAGS.num_tpu_cores,
                        per_host_input_for_training = is_per_host)
                    )
    model_fn    = self._model_fn_builder(
                    bert_config = self.bert_config
                    )
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    self.train = tfBert.BertEstimator(tf.compat.v1.estimator.tpu.TPUEstimator(
                            use_tpu  = FLAGS.use_tpu,
                            model_fn = model_fn,
                            config   = run_config,
                            params   = None,
                            train_batch_size   = self.train_batch_size,
                            eval_batch_size    = self.eval_batch_size,
                            ),
                 data_generator
                 )
    l.getLogger().info(self.GetShortSummary())
    return

  def _ConfigSampleParams(self,
                          data_generator: data_generators.MaskLMBatchGenerator,
                          sampler: samplers.Sampler,
                          ) -> None:
    """
    Model parameter initialization for inference.
    """
    if self.bert_config is None:
      self._ConfigModelParams()
    self.sampler = sampler

    if sampler.sequence_length > self.bertAttrs['max_position_embeddings']:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (sampler.sequence_length, self.bertAttrs['max_position_embeddings']))
      
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.compat.v1.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project)

    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config  = tf.compat.v1.estimator.tpu.RunConfig(
        cluster    = tpu_cluster_resolver,
        master     = FLAGS.master,
        model_dir  = str(self.ckpt_path),
        tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
            num_shards = FLAGS.num_tpu_cores,
            per_host_input_for_training = is_per_host))

    model_fn = self._model_fn_builder(bert_config = self.bert_config)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    self.sample = tfBert.BertEstimator(tf.compat.v1.estimator.tpu.TPUEstimator(
                            use_tpu  = FLAGS.use_tpu,
                            model_fn = model_fn,
                            config   = run_config,
                            params   = {'sampling_temperature': sampler.temperature},
                            predict_batch_size = sampler.batch_size
                            ),
                  data_generator
                  )
    l.getLogger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  @property
  def is_trained(self):
    if FLAGS.select_checkpoint_step >= 0:
      return True
    else:
      for file_path in self.ckpt_path.iterdir():
        filename = file_path.stem
        if "model.ckpt-" in filename:
          step_ckpt = int(filename.replace("model.ckpt-", ""))
          if step_ckpt >= self.num_train_steps:
            return True
    return False  

  def Train(self,
            corpus,
            test_sampler: typing.Optional[samplers.Sampler] = None,
            **unused_kwargs
            ) -> None:

    del unused_kwargs

    if self.train is None:

      data_generator = data_generators.MaskLMBatchGenerator.TrainMaskLMBatchGenerator(
                         corpus, self.config.training, self.cache.path)
      self._ConfigTrainParams(data_generator)

    if not self.is_trained:

      train_input_fn = self.train.data_generator.generateTfDataset(
          sequence_length = self.config.training.sequence_length,
          num_cpu_threads = os.cpu_count(),
          use_tpu = FLAGS.use_tpu,
          is_training = True)

      l.getLogger().info("Splitting {} steps into {} equivalent epochs, {} steps each. Rejected {} redundant step(s)".format(
                                        self.num_train_steps, self.num_epochs, 
                                        self.steps_per_epoch, self.config.training.num_train_steps - self.num_train_steps
                                        )
                        )
      if FLAGS.sample_per_epoch == 0:
        self.train.estimator.train(input_fn = train_input_fn, max_steps = self.num_train_steps)
      else:
        sampler, observers = self._getTestSampler(test_sampler, self.config.training.sequence_length)
        self.InitSampling(sampler, self.config.training.random_seed)
        for ep in range(self.num_epochs):
          self.train.estimator.train(input_fn = train_input_fn, steps = self.steps_per_epoch)
          for _ in range(FLAGS.sample_per_epoch):
            start_time   = datetime.datetime.utcnow()
            self.InitSampleBatch()
            sample_batch = self.SampleNextIndices()
            end_time     = datetime.datetime.utcnow()
            for sample in sample_batch:
              sample_proto = model_pb2.Sample(
                train_step                = (ep + 1) * self.steps_per_epoch,
                sampler_feed              = sampler.start_text,
                text                      = self.atomizer.DeatomizeIndices(sample, ignore_token = self.atomizer.padToken).replace("\\n", "\n"),
                encoded_text              = ",".join([str(t) for t in sample]),
                sample_time_ms            = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
                num_tokens                = len(sample),
                date_added                = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
              )
              for obs in observers:
                obs.OnSample(sample_proto)
      if not FLAGS.force_eval:
        self.Validate()
  
    if FLAGS.force_eval and not self.is_validated:
      self.Validate()
    self.telemetry.TfRecordEpochs()
    return

  def Validate(self) -> None:
    l.getLogger().info("BERT Validation")
    eval_input_fn = self.train.data_generator.generateTfDataset(
        sequence_length = self.config.training.sequence_length,
        num_cpu_threads = os.cpu_count(),
        is_training     = False)

    result = self.train.estimator.evaluate(input_fn=eval_input_fn, steps=self.max_eval_steps)
    self._writeValidation(result)
    self.is_validated = True
    return

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    if self.sample is None or sampler.hash != self.sampler.hash:
      data_generator = data_generators.MaskLMBatchGenerator.SampleMaskLMBatchGenerator(
                         sampler, self.atomizer, seed, self.config.architecture.max_position_embeddings
                       )
      self._ConfigSampleParams(data_generator, sampler)

    ## TODO save that stuff somewhere
    l.getLogger().info("Initialized model samples in {}".format(self.sample_path))
    return 

  def InitSampleBatch(self, *unused_args, **unused_kwargs) -> None:
    """Batch-specific initialization. Called once when a new batch is going to be generated"""
    del unused_args
    del unused_kwargs
    self.sample.data_generator.InitSampleBatch()
    return 

  def SampleNextIndices(self, *unused_args, **unused_kwargs):
    """Called iteratively to build a single batch of samples, until termination criteria stops calling"""
    del unused_kwargs
    del unused_args
    if self.sample is None:
      raise ValueError("Bert sampler has not been initialized.")

    predict_input_fn  = self.sample.data_generator.generateTfSamples()
    predict_generator = self.sample.estimator.predict(input_fn = predict_input_fn)

    output_seq, done = None, False
    for step in predict_generator:
      output_seq, done = self.sample.data_generator.updateSampleBatch(
        step['input_ids'], step['masked_lm_predictions']
        )
    return output_seq

  def _getTestSampler(self, test_sampler, sequence_length):
    if test_sampler is None:
      sampler_str = [
          "start_text: \"kernel void A(const double g, const double i){\\n  [HOLE] = [HOLE]\\n  int a = g + [HOLE]\"",
          "batch_size: 2",
          "sequence_length: {}".format(sequence_length),
          "temperature_micros: 800000",
      ]
      mock_config = pbutil.FromString('\n'.join(sampler_str), sampler_pb2.Sampler())
      sampler = samplers.Sampler(mock_config, sample_db_name = "epoch_samples.db")
    else:
      sampler = test_sampler
    sampler.Specialize(self.atomizer)
    observers = [sample_observers.PrintSampleObserver()]
    if FLAGS.store_samples_db:
      observers.append(sample_observers.SamplesDatabaseObserver(
        "sqlite:///{}".format(self.sample_path / sampler.hash / sampler.sample_db_name)
        )
      )
      sampler.symlinkModelDB(
        self.sample_path / sampler.hash,
        self.hash
      )
    return sampler, observers

  def GetShortSummary(self) -> str:
    l.getLogger().debug("deeplearning.clgen.models.tf_bert.tfBert.GetShortSummary()")
    return (
      f"h_s: {self.config.architecture.hidden_size}, "
      f"#h_l: {self.config.architecture.num_hidden_layers}, "
      f"#att_h: {self.config.architecture.num_attention_heads}, "
      f"imd_s: {self.config.architecture.intermediate_size}, "
      f"h_act: {self.config.architecture.hidden_act}, "
      f"{model_pb2.NetworkArchitecture.Backend.Name(self.config.architecture.backend)} "
      "network"
      "\n"
      # self.data_generator.GetShortSummary() # TODO
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
    paths += [ path.absolute() for path in (self.cache.path / "samples").iterdir() ]
    # paths += self.data_generator.InferenceManifest # TODO
    return sorted(paths)

  def _writeValidation(self,
                       result
                       ) -> None:
    with tf.io.gfile.GFile(self.validation_results_path, "w") as writer:
      l.getLogger().info("Validation set result summary")
      for key in sorted(result.keys()):
        l.getLogger().info("{}: {}".format(key, str(result[key])))
        writer.write("{}: {}\n".format(key, str(result[key])))
    return 

  def _model_fn_builder(self,
                      bert_config, 
                      ):
    """Returns `model_fn` closure for TPUEstimator."""

    def _model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
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
          input_mask=input_mask,
          token_type_ids=None, # You can ignore. Used for double sentences (sA -> 0, sB ->1). Now all will be zero
          use_one_hot_embeddings=FLAGS.use_tpu)

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
      if (self.ckpt_path / "checkpoint").exists():
        (assignment_map, initialized_variable_names
        ) = model.get_assignment_map_from_checkpoint(tvars, str(self.ckpt_path))
        if FLAGS.use_tpu:

          def _tpu_scaffold():
            tf.compat.v1.train.init_from_checkpoint(str(self.ckpt_path), assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = _tpu_scaffold
        else:
          if mode != tf.compat.v1.estimator.ModeKeys.PREDICT:
            l.getLogger().info("Loading model checkpoint from: {}".format(str(self.ckpt_path)))
          tf.compat.v1.train.init_from_checkpoint(str(self.ckpt_path), assignment_map)

      output_spec = None
      if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:        
        with tf.compat.v1.variable_scope("training"):

          train_op = optimizer.create_optimizer(
              total_loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, FLAGS.use_tpu)

          training_hooks = self.GetTrainingHooks(tensors = {'Loss': total_loss},
                                                masked_lm_loss = masked_lm_loss,
                                                next_sentence_loss = next_sentence_loss,
                                                total_loss = total_loss,
                                                learning_rate = self.learning_rate,
                                                )

          output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
              mode = mode,
              loss = total_loss,
              train_op = train_op,
              training_hooks = training_hooks,
              scaffold_fn = scaffold_fn)
      elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:
        with tf.compat.v1.variable_scope("evaluation"):

          def _metric_fn(masked_lm_example_loss, masked_lm_predictions, masked_lm_ids,
                        masked_lm_weights, next_sentence_example_loss,
                        next_sentence_predictions, next_sentence_labels):
            """Computes the loss and accuracy of the model."""            
            masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                labels=masked_lm_ids,
                predictions=masked_lm_predictions,
                weights=masked_lm_weights, 
                name = "masked_lm_mean_loss")
            masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                values=masked_lm_example_loss, 
                weights=masked_lm_weights, 
                name = "masked_lm_mean_loss")

            next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
            next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
                labels=next_sentence_labels, 
                predictions=next_sentence_predictions, 
                name = "next_sentence_accuracy")
            next_sentence_mean_loss = tf.compat.v1.metrics.mean(
                values=next_sentence_example_loss, 
                name = "next_sentence_mean_loss")

            return {
                # 'input_ids'                 : input_ids,
                # 'masked_lm_predictions'     : masked_lm_predictions,
                # 'next_sentence_predictions' : next_sentence_predictions,
                'masked_lm_accuracy'        : masked_lm_accuracy,
                'masked_lm_loss'            : masked_lm_mean_loss,
                'next_sentence_accuracy'    : next_sentence_accuracy,
                'next_sentence_loss'        : next_sentence_mean_loss,
            }

          masked_lm_log_probs = tf.reshape(
            masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]]
          )
          masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32,# name = "masked_lm_predictions"
          )
          next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]]
          )
          next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32,# name = "next_sentence_predictions"
          )

          eval_metrics = (_metric_fn, [
              masked_lm_example_loss, masked_lm_predictions, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_predictions, next_sentence_labels
          ])
          evaluation_hooks = self.GetValidationHooks(
            mode = mode, 
            url  = self.logfile_path / "validation_samples.db",
            atomizer                  = self.atomizer,
            input_ids                 = input_ids, 
            input_mask                = input_mask, 
            masked_lm_positions       = masked_lm_positions,
            masked_lm_ids             = masked_lm_ids,
            masked_lm_weights         = masked_lm_weights,
            next_sentence_labels      = next_sentence_labels,
            masked_lm_predictions     = masked_lm_predictions,
            next_sentence_predictions = next_sentence_predictions,
          ) 

          output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
              mode = mode,
              loss = total_loss,
              evaluation_hooks = evaluation_hooks,
              eval_metrics = eval_metrics,
              scaffold_fn = scaffold_fn)
      elif mode == tf.compat.v1.estimator.ModeKeys.PREDICT:

        with tf.compat.v1.variable_scope("predict"):

          mask_batch_size, mask_seq_length = model.get_shape_list(masked_lm_positions,  expected_rank = 2)
          next_batch_size, next_seq_length = model.get_shape_list(next_sentence_labels, expected_rank = 2)

          masked_lm_log_probs       = tf.reshape(masked_lm_log_probs,     [-1, masked_lm_log_probs.shape[-1]])
          next_sentence_log_probs   = tf.reshape(next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])

          if FLAGS.categorical_sampling:

            mlm_sampler = tfp.distributions.Categorical(logits = masked_lm_log_probs / params['sampling_temperature'])
            nsp_sampler = tfp.distributions.Categorical(logits = next_sentence_log_probs / params['sampling_temperature'])

            masked_lm_predictions     = mlm_sampler.sample()
            next_sentence_predictions = nsp_sampler.sample()

          else:

            masked_lm_predictions     = tf.argmax(masked_lm_log_probs,     axis = -1, output_type = tf.int32)
            next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis = -1, output_type = tf.int32)

          masked_lm_predictions     = tf.reshape(masked_lm_predictions,     shape = [mask_batch_size, mask_seq_length])
          next_sentence_predictions = tf.reshape(next_sentence_predictions, shape = [next_batch_size, next_seq_length])

          input_ids                 = tf.expand_dims(input_ids,                 0, name = "input_ids")
          masked_lm_predictions     = tf.expand_dims(masked_lm_predictions,     0, name = "masked_lm_predictions")
          next_sentence_predictions = tf.expand_dims(next_sentence_predictions, 0, name = "next_sentence_predictions")

          prediction_metrics = {
              'input_ids'                 : input_ids,
              'masked_lm_predictions'     : masked_lm_predictions,
              'next_sentence_predictions' : next_sentence_predictions,
          }
          output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
              mode = mode,
              predictions = prediction_metrics,
              scaffold_fn = scaffold_fn)
      else:
        raise ValueError("{} is not a valid mode".format(mode))
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

  def GetTrainingHooks(self,
                       tensors: typing.Dict[str, tf.Tensor],
                       log_steps:  int = None, 
                       max_steps:  int = None,
                       output_dir: str = None,
                       **kwargs
                       ) -> typing.List[tf.estimator.SessionRunHook]:
    if log_steps is None:
      log_steps = self.steps_per_epoch
    if max_steps is None:
      max_steps = self.num_train_steps
    if output_dir is None:
      output_dir = str(self.logfile_path)
    return [
            tf.estimator.SummarySaverHook(save_steps = log_steps,
                                          output_dir = output_dir,
                                          summary_op = [ tf.compat.v1.summary.scalar(name, value) 
                                                          for name, value in kwargs.items()
                                                        ]
                                          ),
            hooks.tfLogTensorHook(tensors = tensors, 
                                  log_steps = log_steps, 
                                  at_end = True
                                  ),
            hooks.tfProgressBar(max_length = max_steps),
           ]
  
  def GetValidationHooks(self,
                         max_steps = None,
                         **kwargs
                         ) -> typing.List[tf.estimator.SessionRunHook]:
    if max_steps is None:
      max_steps = self.max_eval_steps
    return [
            hooks.tfProgressBar(max_length = max_steps, mode = tf.compat.v1.estimator.ModeKeys.EVAL),
            hooks.writeValidationDB(**kwargs)
            ]
