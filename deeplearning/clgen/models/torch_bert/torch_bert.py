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
from deeplearning.clgen import validation_database
from deeplearning.clgen.util.tf import tf
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.models.tf_bert import model
from deeplearning.clgen.models.tf_bert import optimizer
from deeplearning.clgen.models.tf_bert import hooks
from deeplearning.clgen.models.tf_bert.data_generator import MaskLMBatchGenerator

from eupy.native import logger as l

FLAGS = flags.FLAGS

# flags.DEFINE_integer(
#   "select_checkpoint_step",
#   -1,
#   "Select step checkpoint for sample. Re-training with this flag is not supported. "
#   "To restart from specific checkpoint, you have to manually remove the checkpoints after the desired one."
#   "Default: -1, flag ignored and latest checkpoint is loaded."
# )

# flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

# flags.DEFINE_boolean("force_eval", False, "Run Validation no matter what.")

# flags.DEFINE_integer("sample_per_epoch", 3, "Set above zero to sample model after every epoch.")

# flags.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# flags.DEFINE_boolean("mirror_gpus", False, "Set True to distribute training across all system's GPUs. (Only usable when use_tpu is False).")

# flags.DEFINE_boolean("categorical_sampling", True, "Use categorical distribution on logits when sampling.")

# flags.DEFINE_string(
#     "tpu_name", None,
#     "The Cloud TPU to use for training. This should be either the name "
#     "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
#     "url.")

# flags.DEFINE_string(
#     "tpu_zone", None,
#     "[Optional] GCE zone where the Cloud TPU is located in. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string(
#     "gcp_project", None,
#     "[Optional] Project name for the Cloud TPU-enabled project. If not "
#     "specified, we will attempt to automatically detect the GCE project from "
#     "metadata.")

# flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# flags.DEFINE_integer(
#     "num_tpu_cores", 8,
#     "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class torchBert(backends.BackendBase):

  class BertEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT estimator pipeline."""
    estimator      : tf.compat.v1.estimator.tpu.TPUEstimator
    data_generator : MaskLMBatchGenerator

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
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
          "layer_norm_eps"               : self.config.architecture.layer_norm_eps,
          "pad_token_id"                 : self.atomizer.padToken,
    }
    l.getLogger().error("Issue with TORCH dropout probs: TORCH might not need inverse dropout prob: {}".format(
      self.config.architecture.hidden_dropout_prob
      )
    )
    self.bert_config                     = model.BertConfig.from_dict(self.bertAttrs)
    return

  def _ConfigTrainParams(self, 
                         data_generator: MaskLMBatchGenerator
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
                          data_generator: MaskLMBatchGenerator,
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

  def samplesWithCategorical(self):
    return FLAGS.categorical_sampling

  def Train(self,
            corpus,
            test_sampler: typing.Optional[samplers.Sampler] = None,
            **unused_kwargs
            ) -> None:

    """
    Main training entry point.

    Args:
      model_path (:obj:`str`, `optional`):
        Local path to the model if the model to train has been instantiated from a local path. If present,
        training will resume from the optimizer/scheduler states loaded here.
    """

    train_sampler = self._get_train_sampler()

    train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.args.train_batch_size,
      sampler=train_sampler,
      collate_fn=self.data_collator,
      drop_last=self.args.dataloader_drop_last,
    )

    # if self.args.max_steps > 0:
    #   t_total = self.args.max_steps
    #   num_train_epochs = (
    #     self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
    #   )
    # else:
    #   t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
    #   num_train_epochs = self.args.num_train_epochs
    #   self.args.max_steps = t_total

    self.create_optimizer_and_scheduler(num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
      model_path is not None
      and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
      and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
    ):
      # Load in optimizer and scheduler states
      self.optimizer.load_state_dict(
        torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
      )
      self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    model = self.model
    # if self.args.fp16 and _use_apex:
    #   if not is_apex_available():
    #     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #   model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if self.args.n_gpu > 1:
      model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if self.args.local_rank != -1:
      model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[self.args.local_rank],
        output_device=self.args.local_rank,
        find_unused_parameters=True,
      )

    # if self.tb_writer is not None:
    #   self.tb_writer.add_text("args", self.args.to_json_string())
    #   self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

    # Train!
    if is_torch_tpu_available():
      total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
    else:
      total_train_batch_size = (
        self.args.train_batch_size
        * self.args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
      )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", self.num_examples(train_dataloader))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    self.global_step = 0
    self.epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if model_path is not None:
      # set global_step to global_step of last saved checkpoint from model path
      try:
        self.global_step = int(model_path.split("-")[-1].split("/")[0])
        epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = self.global_step % (
          len(train_dataloader) // self.args.gradient_accumulation_steps
        )

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", self.global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
      except ValueError:
        self.global_step = 0
        logger.info("  Starting fine-tuning.")

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
      epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=not self.is_local_process_zero()
    )
    for epoch in train_iterator:
      if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
        train_dataloader.sampler.set_epoch(epoch)

      if is_torch_tpu_available():
        parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
          self.args.device
        )
        epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
      else:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_process_zero())

      # Reset the past mems state at the beginning of each epoch if necessary.
      if self.args.past_index >= 0:
        self._past = None

      for step, inputs in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
          steps_trained_in_current_epoch -= 1
          continue

        tr_loss += self.training_step(model, inputs)

        if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
          # last step in epoch but step is always smaller than gradient_accumulation_steps
          len(epoch_iterator) <= self.args.gradient_accumulation_steps
          and (step + 1) == len(epoch_iterator)
        ):
          if self.args.fp16 and _use_native_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
          elif self.args.fp16 and _use_apex:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
          else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

          if is_torch_tpu_available():
            xm.optimizer_step(self.optimizer)
          elif self.args.fp16 and _use_native_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
          else:
            self.optimizer.step()

          self.lr_scheduler.step()
          model.zero_grad()
          self.global_step += 1
          self.epoch = epoch + (step + 1) / len(epoch_iterator)

          if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
            self.global_step == 1 and self.args.logging_first_step
          ):
            logs: Dict[str, float] = {}
            logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
              self.lr_scheduler.get_last_lr()[0]
              if version.parse(torch.__version__) >= version.parse("1.4")
              else self.lr_scheduler.get_lr()[0]
            )
            logging_loss = tr_loss

            self.log(logs)

          if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
            self.evaluate()

          if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
            # In all cases (even distributed/parallel), self.model is always a reference
            # to the model we want to save.
            if hasattr(model, "module"):
              assert (
                model.module is self.model
              ), f"Module {model.module} should be a reference to self.model"
            else:
              assert model is self.model, f"Model {model} should be a reference to self.model"
            # Save model checkpoint
            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

            self.save_model(output_dir)

            if self.is_world_process_zero():
              self._rotate_checkpoints()

            if is_torch_tpu_available():
              xm.rendezvous("saving_optimizer_states")
              xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            elif self.is_world_process_zero():
              torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
          epoch_iterator.close()
          break
      if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
        train_iterator.close()
        break
      if self.args.tpu_metrics_debug or self.args.debug:
        if is_torch_tpu_available():
          # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
          xm.master_print(met.metrics_report())
        else:
          logger.warning(
            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
            "configured. Check your training configuration if this is unexpected."
          )

    if self.tb_writer:
      self.tb_writer.close()
    if self.args.past_index and hasattr(self, "_past"):
      # Clean the state at the end of training
      delattr(self, "_past")

    return TrainOutput(self.global_step, tr_loss / self.global_step)

  def Validate(self) -> None:
    l.getLogger().info("BERT Validation")
    if self.max_eval_steps <= 0:
      return
    for tf_set in self.train.data_generator.dataset:
      tf_set_paths = self.train.data_generator.dataset[tf_set]['tf_record']
      l.getLogger().info("BERT Validation on {}".format(', '.join([pathlib.Path(x).stem for x in tf_set_paths])))
      eval_input_fn = self.train.data_generator.generateTfDataset(
          sequence_length = self.config.training.sequence_length,
          num_cpu_threads = os.cpu_count(),
          is_training     = False,
          eval_set        = tf_set_paths
          )
      result = self.train.estimator.evaluate(input_fn=eval_input_fn, steps=self.max_eval_steps)
      self._writeValidation(result, tf_set)
    self.is_validated = True
    return

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    data_generator = MaskLMBatchGenerator.SampleMaskLMBatchGenerator(
                       sampler, self.atomizer, seed, self.config.architecture.max_position_embeddings, self.cache.path
                     )
    self._ConfigSampleParams(data_generator, sampler)
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
      output_seq, sampleIndices = self.sample.data_generator.updateSampleBatch(
        step['input_ids'], step['masked_lm_predictions']
        )
    return output_seq, sampleIndices

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
    if sampler.isFixedStr:
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

  def _writeValidation(self, result, tf_set) -> None:
    with tf.io.gfile.GFile(self.validation_results_path, "w") as writer:
      db = validation_database.ValidationDatabase("sqlite:///{}".format(str(self.logfile_path / "validation_samples.db")))
      r = [ "{}: {}".format(key, str(result[key])) for key in result.keys() ]
      with db.Session(commit = True) as session:
        exists = session.query(validation_database.ValResults.key).filter_by(key = str(tf_set)).scalar() is not None
        if exists:
          entry = session.query(validation_database.ValResults).filter_by(key = str(tf_set)).first()
          entry.results = "\n".join(r)
        else:
          session.add(validation_database.ValResults(key = str(tf_set), results = "\n".join(r)))
    return 
