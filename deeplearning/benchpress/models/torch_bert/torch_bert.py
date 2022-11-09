# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BenchPress language model training and sampling wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import shutil
import humanize
import typing
import pathlib
import datetime
import time
import numpy as np
from absl import flags
import tqdm
from collections import OrderedDict


from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.samplers import samplers
from deeplearning.benchpress.samplers import sample_observers
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import plotter
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.proto import sampler_pb2
from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.models import backends
from deeplearning.benchpress.models import telemetry
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.models.torch_bert import model
from deeplearning.benchpress.models.torch_bert import config
from deeplearning.benchpress.models.torch_bert import optimizer
from deeplearning.benchpress.models.torch_bert import hooks
from deeplearning.benchpress.models.torch_bert.data_generator import torchLMDataGenerator

from deeplearning.benchpress.util import logging as l
from eupy.hermes import client

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "reward_compilation",
  -1,
  "Select to integrate LLVM compiler into training regime."
  "During training, the target token will be asked to fill the first token of the hole."
  "If this flag is selected to True, the model will fill entirely the hole, as in inference."
  "The fully generated sample will be checked for syntactic correctness with LLVM."
  "If the sample compiles, then loss will be zero-ed for that instance, hence will be rewarded."
  "[Default: -1]: do not use comp-rewarded training."
  "Any integer >= 0: Kick-in this mode after this training step. 0 uses this method from start."
)

flags.DEFINE_boolean(
  "validate_per_epoch",
  True,
  "Calculate and plot validation loss per end of epoch."
)

flags.DEFINE_integer(
  "eval_steps_per_epoch",
  1000,
  "Set validation steps at the end of epoch for validation loss calculation."
)

flags.DEFINE_boolean(
  "is_trt",
  False,
  "Use TensorRT for the sampling model."
)

class torchBert(backends.BackendBase):

  class BertEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : torchLMDataGenerator
    optimizer      : typing.Any
    scheduler      : typing.Any

  class SampleBertEstimator(typing.NamedTuple):
    """Named tuple for sampling BERT."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : torchLMDataGenerator

  @property
  def hidden_state_size(self):
    return self.config.architecture.hidden_size

  def __repr__(self):
    return "BenchPress"

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
    from deeplearning.benchpress.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    if self.config.architecture.HasField("feature_encoder") and self.config.architecture.feature_encoder:
      self.feature_encoder   = True
      self.feature_tokenizer = tokenizers.FeatureTokenizer.FromArgs(
        self.config.architecture.feature_singular_token_thr,
        self.config.architecture.feature_max_value_token,
        self.config.architecture.feature_token_range
      )
      self.feature_sequence_length = self.config.architecture.feature_sequence_length
    else:
      self.feature_encoder         = False
      self.feature_tokenizer       = None
      self.feature_sequence_length = None

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.bertAttrs         = {}
    self.featureAttrs      = {}
    self.bert_config       = None

    self.train             = None
    self.sample            = None
    self.predict_generator = None
    self.sampler           = None

    self.train_batch_size  = None
    self.eval_batch_size   = None
    self.learning_rate     = None
    self.num_train_steps   = None

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"

    self.logfile_path      = self.cache.path / "logs"
    if self.config.HasField("pre_train_corpus"):
      self.pre_logfile_path = self.logfile_path / "pre_train"

    self.telemetry         = telemetry.TrainingLogger(self.logfile_path)
    if self.config.HasField("pre_train_corpus"):
      self.pre_telemetry   = telemetry.TrainingLogger(self.logfile_path / "pre_train")

    self.is_validated      = False
    self.trained           = False
    l.logger().info("BERT Model config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, is_sampling):
    """General model hyperparameters initialization."""
    self.bertAttrs = {
          "vocab_size"                   : self.tokenizer.vocab_size,
          "hidden_size"                  : self.config.architecture.hidden_size,
          "num_hidden_layers"            : self.config.architecture.num_hidden_layers,
          "num_attention_heads"          : self.config.architecture.num_attention_heads,
          "intermediate_size"            : self.config.architecture.intermediate_size,
          "hidden_act"                   : self.config.architecture.hidden_act,
          "hidden_dropout_prob"          : self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob" : self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"      : self.config.architecture.max_position_embeddings,
          "type_vocab_size"              : self.config.architecture.type_vocab_size,
          "initializer_range"            : self.config.architecture.initializer_range,
          "layer_norm_eps"               : self.config.architecture.layer_norm_eps,
          "pad_token_id"                 : self.tokenizer.padToken,
    }
    if self.feature_encoder:
      self.featureAttrs = {
        "feature_encoder"                 : self.feature_encoder,
        "feature_sequence_length"         : self.feature_sequence_length,
        "feature_embedding_size"          : self.config.architecture.feature_embedding_size,
        "feature_pad_idx"                 : self.feature_tokenizer.padToken,
        "feature_dropout_prob"            : self.config.architecture.feature_dropout_prob,
        "feature_vocab_size"              : len(self.feature_tokenizer),
        "feature_num_attention_heads"     : self.config.architecture.feature_num_attention_heads,
        "feature_transformer_feedforward" : self.config.architecture.feature_transformer_feedforward,
        "feature_layer_norm_eps"          : self.config.architecture.feature_layer_norm_eps,
        "feature_num_hidden_layers"       : self.config.architecture.feature_num_hidden_layers,
      }
    self.bert_config = config.BertConfig.from_dict(
      self.bertAttrs,
      **self.featureAttrs,
      xla_device         = self.torch_tpu_available,
      reward_compilation = FLAGS.reward_compilation,
      is_sampling        = is_sampling,
    )
    return

  def _ConfigTrainParams(self, 
                         data_generator: torchLMDataGenerator,
                         pre_train: bool,
                        ) -> None:
    """
    Model parameter initialization for training and validation.
    """
    self._ConfigModelParams(is_sampling = False)

    self.train_batch_size                 = self.config.training.batch_size
    self.eval_batch_size                  = self.config.training.batch_size
    self.learning_rate                    = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_warmup_steps                 = self.config.training.num_warmup_steps if not pre_train else self.config.training.num_prewarmup_steps
    self.max_grad_norm                    = 1.0

    self.steps_per_epoch                  = data_generator.steps_per_epoch
    self.current_step                     = None
    self.num_epochs                       = data_generator.num_epochs
    self.num_train_steps                  = self.steps_per_epoch * self.num_epochs
    self.max_eval_steps                   = FLAGS.max_eval_steps

    self.validation_results_file          = "val_results.txt"
    self.validation_results_path          = os.path.join(str(self.logfile_path if not pre_train else self.pre_logfile_path), self.validation_results_file)

    m = model.BertForPreTraining(
          self.bert_config,
          tokenizer = self.tokenizer,
          target_lm = "hole" if self.config.training.data_generator.HasField("hole") else "mask"
        ).to(self.pytorch.offset_device)

    if self.pytorch.num_nodes > 1:
      distrib.barrier()
      m = self.torch.nn.parallel.DistributedDataParallel(
        m,
        device_ids    = [self.pytorch.offset_device],
        output_device = self.pytorch.offset_device,
        find_unused_parameters = True,
      )
    elif self.pytorch.num_gpus > 1:
      m = self.torch.nn.DataParallel(m)

    opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
      model           = m,
      num_train_steps = self.num_train_steps,
      warmup_steps    = self.num_warmup_steps,
      learning_rate   = self.learning_rate,
    )

    self.train = torchBert.BertEstimator(
                  m, data_generator, opt, lr_scheduler
                )
    l.logger().info(self.GetShortSummary())
    return

  def _ConfigSampleParams(self,
                          data_generator: torchLMDataGenerator,
                          sampler: samplers.Sampler,
                          ) -> None:
    """
    Model parameter initialization for inference.
    """
    self._ConfigModelParams(is_sampling = True)
    self.sampler = sampler
    self.temperature = sampler.temperature

    if sampler.sequence_length > self.bertAttrs['max_position_embeddings']:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (sampler.sequence_length, self.bertAttrs['max_position_embeddings']))

    if FLAGS.is_trt:
      mdl = model.BertForPreTrainingTRT
    else:
      mdl = model.BertForPreTraining

    m = mdl(
          self.bert_config,
          tokenizer       = self.tokenizer,
          use_categorical = FLAGS.categorical_sampling,
          temperature     = self.temperature,
          target_lm       = "hole" if self.config.training.data_generator.HasField("hole") else "mask"
        ).to(self.pytorch.offset_device)

    if self.pytorch.num_nodes > 1:
      m = self.torch.nn.parallel.DistributedDataParallel(
        m,
        device_ids = [self.pytorch.offset_device],
        output_device = self.pytorch.offset_device,
        find_unused_parameters = True,
      )
    elif self.pytorch.num_gpus > 1:
      m = self.torch.nn.DataParallel(m)

    if FLAGS.is_trt:
      for mdl_instance, dev in zip(m, d):
        mdl_instance.init_engine(self.cache, dev.index, sampler.batch_size, sampler.sequence_length, self.tokenizer.vocab_size, self.config.architecture.max_position_embeddings)

    self.sample = torchBert.SampleBertEstimator(m, data_generator)
    l.logger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  def samplesWithCategorical(self):
    return FLAGS.categorical_sampling

  def GetEncoderModule(self,
                       with_checkpoint    : bool = False,
                       without_label_head : bool = True,
                       **kwargs,
                       ) -> 'torch.nn.Module':
    """Initialize BERT as decoder."""
    attrs = copy.copy(self.bertAttrs)
    if not with_checkpoint:
      attrs = {
        k: v for k, v in kwargs.items()
      }
    elif len(kwargs.keys()) > 0:
      l.logger().warn("Encoder module with_checkpoint will not override max position embeddings, pad and vocab size!")
    generic_config = config.BertConfig.from_dict(
      attrs,
      # **self.featureAttrs,
      xla_device         = self.torch_tpu_available,
      reward_compilation = -1,
      # This is hard-coded to True to allow compile sampler to be initialized. This does not prohibit proper re-train.
      is_sampling        = False,
    )
    m = model.BertForPreTraining(
          generic_config,
          tokenizer  = self.tokenizer,
          target_lm  = "hole" if self.config.training.data_generator.HasField("hole") else "mask",
          without_label_head = without_label_head,
        )
    if with_checkpoint:
      temp_estimator = torchBert.SampleBertEstimator(m, None)
      self.loadCheckpoint(temp_estimator)
      return temp_estimator.model
    else:
      return m

  def GetDecoderModule(self,
                       with_checkpoint    : bool = False,
                       without_label_head : bool = False,
                       **kwargs,
                       ) -> 'torch.nn.Module':
    """Return internal BERT auto-encoder module."""
    attrs = copy.copy(self.bertAttrs)
    if not with_checkpoint:
      attrs = {
        k: v for k, v in kwargs.items()
      }
    elif len(kwargs.keys()) > 0:
      l.logger().warn("Decoder module with_checkpoint will not override max position embeddings, pad and vocab size!")
    generic_config = config.BertConfig.from_dict(
      attrs,
      xla_device          = self.torch_tpu_available,
      reward_compilation  = -1,
      # This is hard-coded to True to allow compile sampler to be initialized. This does not prohibit proper re-train.
      is_sampling         = False,
      is_decoder          = True,
      add_cross_attention = True,
    )
    m = copy.deepcopy(model.BertForPreTraining(
          generic_config,
          tokenizer  = self.tokenizer,
          target_lm  = "hole" if self.config.training.data_generator.HasField("hole") else "mask",
          without_label_head = without_label_head,
        ))
    if with_checkpoint:
      temp_estimator = torchBert.SampleBertEstimator(m, None)
      self.loadCheckpoint(temp_estimator, without_label_head = without_label_head, is_decoder = True)
      return temp_estimator.model
    else:
      return m

  def to_device(self, inputs) -> 'torch.Tensor':
    """
    Move input tensors to torch device and return them.
    """
    inputs['input_ids']            = inputs['input_ids'].to(self.pytorch.device)
    inputs['input_mask']           = inputs['input_mask'].to(self.pytorch.device)
    inputs['position_ids']         = inputs['position_ids'].to(self.pytorch.device)
    inputs['mask_labels']          = inputs['mask_labels'].to(self.pytorch.device)
    if 'input_features' in inputs:
      inputs['input_features'] = inputs['input_features'].to(self.pytorch.device)
    else:
      inputs['input_features'] = None
    return inputs

  def model_step(self,
                 model         : 'torch.nn.Module',
                 inputs        : typing.Dict[str, 'torch.Tensor'],
                 is_validation : bool = False,
                 step          : int  = -1,
                 extract_hidden_state: bool = False,
                 ) -> typing.Dict[str, 'torch.Tensor']:
    """
    Perform a training step on a batch of inputs.
    """
    outputs = model(
                input_ids            = inputs['input_ids'],
                attention_mask       = inputs['input_mask'],
                position_ids         = inputs['position_ids'],
                input_features       = inputs['input_features'],
                masked_lm_labels     = inputs['mask_labels'],
                is_validation        = is_validation,
                step                 = step,
                extract_hidden_state = extract_hidden_state,
              )
    return outputs

  def sample_model_step(self,
                        model                : typing.List['torch.nn.Module'],
                        inputs               : typing.Dict[str, 'torch.Tensor'],
                        iteration            : int = None,
                        extract_hidden_state : bool = False,
                        ) -> typing.Dict[str, typing.List[typing.List[int]]]:
    """
    Specialized forward function.
    Dispatches model replicas across all GPUs, one process each.

    Inputs must be three-dimensional:
    workload_size x batch_size x sequence_length
    """
    start = time.time()
    outputs = {
      'generated_samples': [], 'sample_indices': [],
      'input_ids': [], 'masked_lm_lengths': []
    }
    if extract_hidden_state:
      outputs['hidden_state'] = []

    if iteration is not None:
      desc = "Sampling iteration: {}".format(iteration)
    else:
      desc = "Sampling"

    wload_size = len(inputs['input_ids']) * len(inputs['input_ids'][0])
    inputs = self.to_device(inputs)
    if environment.WORLD_RANK == 0:
      bar = tqdm.auto.trange(wload_size, desc=desc, leave = False, position = 0)
    samples, sample_indices, hidden_state = model(
      workload = (
        inputs['input_ids'],
        inputs['input_mask'],
        inputs['position_ids'],
        inputs['input_features'],
      ),
      bar = bar if environment.WORLD_RANK == 0 else None,
      extract_hidden_state = extract_hidden_state,
    )

    outputs['generated_samples'] = samples.detach()
    outputs['sample_indices']    = sample_indices.detach()
    outputs['input_ids']         = self.torch.reshape(inputs['input_ids'], tuple(samples.shape))
    outputs['masked_lm_lengths'] = self.torch.reshape(inputs['masked_lm_lengths'].to(self.pytorch.device), (samples.shape[0], -1))
    if extract_hidden_state:
      outputs['extract_hidden_state'] = hidden_state.detach()

    outputs['generated_samples'] = list(outputs['generated_samples'].cpu().numpy())
    outputs['sample_indices']    = list(outputs['sample_indices'].cpu().numpy())
    outputs['input_ids']         = list(outputs['input_ids'].cpu().numpy())
    outputs['masked_lm_lengths'] = list(outputs['masked_lm_lengths'].cpu().numpy())
    if extract_hidden_state:
      outputs['hidden_state'] = list(outputs['hidden_state'].cpu().numpy())

    end = time.time()
    return outputs, end-start

  def PreTrain(self,
               corpus,
               test_sampler: typing.Optional[samplers.Sampler] = None,
               **unused_kwargs
               ) -> None:
    """
    Pre-training entry point.
    """
    self.Train(corpus, test_sampler, pre_train = True)
    return

  def Train(self,
            corpus,
            test_sampler : typing.Optional[samplers.Sampler] = None,
            pre_train    : bool = False,
            **unused_kwargs
            ) -> None:
    """
    Main training entry point.
    """

    if FLAGS.only_sample:
      del self.train
      self.train = None
      return

    self._ConfigTrainParams(
      torchLMDataGenerator.TrainMaskLMBatchGenerator(
        corpus, self.config.training,
        self.cache.path,
        self.config.training.num_pretrain_steps if pre_train else None,
        pre_train,
        self.feature_encoder,
        self.feature_tokenizer,
        self.feature_sequence_length,
      ), pre_train
    )

    self.current_step = self.loadCheckpoint(self.train, pre_train = pre_train)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    if self.current_step >= 0:
      l.logger().info("Loaded checkpoint step {}".format(self.current_step))
    self.current_step = max(0, self.current_step)

    if self.current_step < self.num_train_steps:
      self.train.model.zero_grad()

      ## Set batch size in case of TPU training or distributed training.
      if self.torch_tpu_available:
        total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
      else:
        total_train_batch_size = (
          self.train_batch_size
          * (self.torch.distributed.get_world_size() if self.pytorch.num_nodes > 1 else 1)
        )

      # Set dataloader in case of TPU training.
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            self.train.data_generator.dataloader, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)
      else:
        loader = self.train.data_generator.dataloader

      # Get dataloader iterator and setup hooks.
      batch_iterator = iter(loader)
      if self.is_world_process_zero():
        train_hook = hooks.tensorMonitorHook(
          self.logfile_path if not pre_train else self.pre_logfile_path, self.current_step, min(self.steps_per_epoch, FLAGS.monitor_frequency)
        )
      if FLAGS.reward_compilation >= 0 and not pre_train:
        correct_sample_obs = sample_observers.SamplesDatabaseObserver(
          self.logfile_path / "correct_samples.db"
        )
      else:
        correct_sample_obs = None
      
      total_steps = self.config.training.num_pretrain_steps if pre_train else self.config.training.num_train_steps
      l.logger().info(
        "Splitting {} steps into {} equivalent epochs, {} steps each. Rejected {} redundant step(s)".format(
          self.num_train_steps, self.num_epochs, 
          self.steps_per_epoch, total_steps - self.num_train_steps
        )
      )
      try:
        self.train.model.train()
        epoch_iter = tqdm.auto.trange(self.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(self.num_epochs)
        for epoch in epoch_iter:

          # In distributed mode, calling the set_epoch() method at
          # the beginning of each epoch before creating the DataLoader iterator
          # is necessary to make shuffling work properly across multiple epochs.
          # Otherwise, the same ordering will be always used.
          if self.pytorch.num_nodes > 1:
            loader.sampler.set_epoch(epoch)

          if epoch < self.current_step // self.steps_per_epoch:
            continue # Stupid bar won't resume.

          batch_iter = tqdm.auto.trange(self.steps_per_epoch, desc="Batch", leave = False) if self.is_world_process_zero() else range(self.steps_per_epoch)
          for step in batch_iter:
            if self.is_world_process_zero():
              start = datetime.datetime.utcnow()
            try:
              inputs = next(batch_iterator)
            except StopIteration:
              # dataloader has different len() than steps_per_epoch.
              # This is the easiest way to infinite-loop dataloaders in pytorch.
              batch_iterator = iter(loader)
              inputs = next(batch_iterator)

            self.current_step += 1
            # Move inputs to torch device.
            inputs     = self.to_device(inputs)
            # Run model step on batch
            step_out   = self.model_step(self.train.model, inputs, step = epoch * self.steps_per_epoch + step)
            # Collect losses and backpropagate
            total_loss = step_out['total_loss'].mean()
            total_loss.backward()

            self.torch.nn.utils.clip_grad_norm_(self.train.model.parameters(), self.max_grad_norm)
            if self.torch_tpu_available:
              self.pytorch.torch_xla.optimizer_step(self.train.optimizer)
            else:
              self.train.optimizer.step()
            self.train.scheduler.step()

            ## Collect tensors for logging.
            if self.pytorch.num_nodes > 1:
              self.torch.distributed.barrier()
              total_loss         = [self.torch.zeros(tuple(step_out['total_loss'        ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
              masked_lm_loss     = [self.torch.zeros(tuple(step_out['masked_lm_loss'    ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
              # next_sentence_loss = [self.torch.zeros(tuple(step_out['next_sentence_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
              masked_lm_lengths  = [self.torch.zeros(tuple(inputs  ['masked_lm_lengths' ].shape), dtype = self.torch.int64  ).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]

              self.torch.distributed.all_gather(masked_lm_loss,     step_out["masked_lm_loss"])
              # self.torch.distributed.all_gather(next_sentence_loss, step_out["next_sentence_loss"])
              self.torch.distributed.all_gather(masked_lm_lengths,  inputs['masked_lm_lengths'].to(self.pytorch.device))
              self.torch.distributed.all_gather(total_loss,         step_out['total_loss'])
            else:
              total_loss         = step_out['total_loss'        ].unsqueeze(0).cpu()
              masked_lm_loss     = step_out['masked_lm_loss'    ].unsqueeze(0).cpu()
              # next_sentence_loss = step_out['next_sentence_loss'].unsqueeze(0).cpu()
              masked_lm_lengths  = inputs['masked_lm_lengths' ].cpu()

            if self.is_world_process_zero():
              exec_time_ms = int(round((datetime.datetime.utcnow() - start).total_seconds() * 1000))
              if FLAGS.reward_compilation >= 0 and FLAGS.reward_compilation <= epoch * self.steps_per_epoch + step and not pre_train:
                ## Logging when compiler reward is enabled in training.
                ## This is not compatible with using DDP, and basically compiler-rewarded training is deprecated and proven to be wrong and inefficient.
                correct_samples = [(x, y) for en, (x, y) in enumerate(zip(inputs['input_ids'].cpu().numpy(), step_out['generated_samples'].cpu().numpy())) if step_out['compile_status'][en] == 1]
                for s in correct_samples:
                  feature_vector = extractor.ExtractFeatures(self.tokenizer.ArrayToCode(s[1]))
                  correct_sample_obs.OnSample(model_pb2.Sample(
                      train_step             = self.current_step,
                      sample_feed            = self.tokenizer.tokensToString(s[0], ignore_token = self.tokenizer.padToken).replace("\\n", "\n"),
                      text                   = self.tokenizer.tokensToString(s[1], ignore_token = self.tokenizer.padToken).replace("\\n", "\n"),
                      encoded_text           = ",".join([str(t) for t in s[1]]),
                      sample_indices         = '',
                      encoded_sample_indices = '',
                      sample_time_ms         = int(round(exec_time_ms / self.train_batch_size)),
                      feature_vector         = "\n".join(["{}:{}".format(k, v) for (k, v) in feature_vector.items()]),
                      num_tokens             = len([x for x in s[1] if x != self.tokenizer.padToken]),
                      categorical_sampling   = False,
                      compile_status         = True,
                      date_added             = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
                    )
                  )
              if not pre_train:
                ## Fine-tuning logging.
                train_hook.step(
                  masked_lm_loss          = sum([ml.mean().item() for ml in masked_lm_loss]) / len(masked_lm_loss),
                  # next_sentence_loss      = sum([nsl.mean().item() for nsl in next_sentence_loss]) / len(next_sentence_loss),
                  total_loss              = sum([tl.mean().item() for tl in total_loss]) / len(total_loss),
                  learning_rate           = self.train.scheduler.get_last_lr()[0],
                  num_correct_samples     = (correct_sample_obs.sample_id if correct_sample_obs is not None else None),
                  batch_avg_hole_len      = sum([sum([int(l) for l in b if l != -1]) / len([int(l) for l in b if l != -1])
                                                 for b in masked_lm_lengths]) / len(masked_lm_lengths),
                  batch_execution_time_ms = exec_time_ms,
                  time_per_sample_ms      = exec_time_ms / self.train_batch_size,
                )
              else:
                ## Pre-training logging.
                train_hook.step(
                  masked_lm_loss          = sum([ml.mean().item() for ml in masked_lm_loss]) / len(masked_lm_loss),
                  # next_sentence_loss      = sum([nsl.mean().item() for nsl in next_sentence_loss]) / len(next_sentence_loss),
                  total_loss              = sum([tl.mean().item() for tl in total_loss]) / len(total_loss),
                  learning_rate           = self.train.scheduler.get_last_lr()[0],
                  batch_avg_hole_len      = sum([sum([int(l) for l in b if l != -1]) / len([int(l) for l in b if l != -1])
                                                 for b in masked_lm_lengths]) / len(masked_lm_lengths),
                  batch_execution_time_ms = exec_time_ms,
                  time_per_sample_ms      = exec_time_ms / self.train_batch_size,
                )
            self.train.model.zero_grad()
            if self.current_step == 0:
              l.logger().info("Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))

          # End of Epoch
          self.saveCheckpoint(self.train, pre_train)
          if self.is_world_process_zero():
            set_mail = "Epoch {} Loss: {}\n".format(self.current_step // self.steps_per_epoch, train_hook.epoch_loss)
            l.logger().info("Epoch {} Loss: {}".format(self.current_step // self.steps_per_epoch, train_hook.epoch_loss))

          if FLAGS.validate_per_epoch > 0 and self.train.data_generator.config.validation_split > 0:
            val_ml_loss = self.Validate(per_epoch = True, pre_train = pre_train)
            if self.is_world_process_zero():
              train_hook.end_epoch(
              val_masked_lm_loss      = val_ml_loss,
              # val_next_sentence_loss  = val_nsp_loss,
              val_total_loss          = val_ml_loss # + val_nsp_loss,
              )
            set_mail += "Validation Loss: {}\n".format(val_ml_loss)
          elif self.is_world_process_zero():
            train_hook.end_epoch()

          if FLAGS.notify_me:
            client.getClient().send_message("clgen:torch_bert", set_mail)

          if self.torch_tpu_available:
            self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())

          if FLAGS.sample_per_epoch > 0:
            if self.is_world_process_zero():
              sampler, observers = self._getTestSampler(test_sampler, self.config.training.sequence_length)
              self.InitSampling(sampler, self.config.training.random_seed)
              for _ in range(FLAGS.sample_per_epoch):
                start_time   = datetime.datetime.utcnow()
                self.InitSampleBatch(sampler)
                org_inputs, input_ids, samples, indices = self.SampleNextIndices()
                end_time = datetime.datetime.utcnow()
                for org, inp, sample, idxs in zip(org_inputs, input_ids, samples, indices):
                  try:
                    stdout = opencl.Compile(self.tokenizer.ArrayToCode(sample))
                    compile_flag = 1
                  except ValueError:
                    compile_flag = 0

                  feature_vector = extractor.ExtractFeatures(self.tokenizer.ArrayToCode(sample))
                  sample_proto = model_pb2.Sample(
                    train_step             = self.current_step,
                    sample_feed            = sampler.start_text,
                    original_input         = self.tokenizer.tokensToString(org,    with_formatting = True, ignore_token = self.tokenizer.padToken),
                    text                   = self.tokenizer.tokensToString(sample, with_formatting = True, ignore_token = self.tokenizer.padToken).replace("\\n", "\n"),
                    encoded_text           = ",".join([str(t) for t in sample]),
                    sample_indices         = ','.join([self.tokenizer.decoder[idx].replace('\n', '\\n') for idx in idxs]).replace('\n', '\\n'),
                    encoded_sample_indices = ','.join([str(idx) for idx in idxs]),
                    sample_time_ms         = int(round(1000 * ((end_time - start_time) / sampler.batch_size).total_seconds())),
                    feature_vector         = "\n".join(["{}:{}".format(k, v) for (k, v) in feature_vector.items()]),
                    num_tokens             = len(sample),
                    compile_status         = compile_flag,
                    categorical_sampling   = self.samplesWithCategorical(),
                    date_added             = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
                  )
                  for obs in observers:
                    obs.OnSample(sample_proto)
            distrib.barrier()      
      except KeyboardInterrupt:
        pass

      if not FLAGS.force_eval:
        _ = self.Validate(pre_train = pre_train)

    if FLAGS.force_eval and not self.is_validated:
      _ = self.Validate(pre_train = pre_train)
    del self.train
    self.train = None
    return

  def TrainBatch(self, inputs) -> None:
    raise NotImplementedError
    return

  def Validate(self, per_epoch = False, pre_train = False) -> float:
    """
    Validation function for torch BERT.

    Arguments:
      per_epoch: Set True if is called at the end of (pre)training epoch.
                 If true, no analytical results are appended to database.
                 Instead, only loss is monitored and plotted.
    """

    if ( (per_epoch and FLAGS.eval_steps_per_epoch <= 0)
      or (not per_epoch and FLAGS.max_eval_steps <= 0)
      or self.config.training.data_generator.validation_split == 0):
      l.logger().info("Skipping BERT Validation.")
      return None, None

    avg_mask_loss = []
    avg_nsp_loss  = []
    preds       = None
    label_ids   = None
    self.train.model.eval()

    for set_idx, (set_name, dataloader) in enumerate(self.train.data_generator.eval_dataloaders()):
      l.logger().info("BERT Validation on {}".format(set_name))
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                          dataloader, [self.pytorch.device]
                    ).per_device_loader(self.pytorch.device)
      else:
        loader = dataloader

      if self.pytorch.num_nodes > 1:
        loader.sampler.set_epoch(set_idx)

      if not per_epoch and self.is_world_process_zero():
        val_hook = hooks.validationSampleHook(
          url = "sqlite:///{}".format(str((self.logfile_path if not pre_train else self.pre_logfile_path) / "validation_samples.db")),
          tokenizer = self.tokenizer,
          model_step = self.current_step
        )
      eval_iterator = iter(loader)
      eval_steps = FLAGS.max_eval_steps if not per_epoch else FLAGS.eval_steps_per_epoch

      try:
        eval_iter = tqdm.auto.trange(self.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(self.num_epochs)
        for step in eval_iter:
          try:
            inputs = next(eval_iterator)
          except StopIteration:
            eval_iterator = iter(loader)
            inputs = next(eval_iterator)

          inputs = self.to_device(inputs)
          with self.torch.no_grad():
            step_out = self.model_step(self.train.model, inputs, is_validation = True)

          if not per_epoch and self.is_world_process_zero():
            val_hook.step(inputs, step_out)

          if self.pytorch.num_nodes > 1:
            self.torch.distributed.barrier()
            masked_lm_loss     = [self.torch.zeros(tuple(step_out['masked_lm_loss'    ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
            # next_sentence_loss = [self.torch.zeros(tuple(step_out['next_sentence_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
            self.torch.distributed.all_gather(masked_lm_loss,     step_out["masked_lm_loss"])
            # self.torch.distributed.all_gather(next_sentence_loss, step_out["next_sentence_loss"])
          else:
            masked_lm_loss     = step_out['masked_lm_loss'    ].cpu()
            # next_sentence_loss = step_out['next_sentence_loss'].cpu()

          avg_mlm_loss = [x.mean().item() for x in masked_lm_loss]
          avg_mask_loss.append(sum(avg_mlm_loss) / len(avg_mlm_loss))
          # avg_nsp_loss.append(next_sentence_loss.mean().item())
      except KeyboardInterrupt:
        pass

      if self.is_world_process_zero() and avg_mask_loss and not per_epoch:
        val_hook.final(set_name, sum(avg_mask_loss) / len(avg_mask_loss))
      if self.pytorch.torch_tpu_available:
        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        self.pytorch.torch_xla_model.master_print(self.pytorch.torch_xla_met.metrics_report())

    if not per_epoch:
      self.is_validated = True
    try:
      return sum(avg_mask_loss) / len(avg_mask_loss)
    except ZeroDivisionError:
      return float('inf'), float('inf')

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None,
                   corpus = None,
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    sample_batch_size = sampler.batch_size
    if self.pytorch.num_nodes == 1 and self.pytorch.num_gpus > 1 and sample_batch_size < self.pytorch.num_gpus:
      l.logger().warn("Sampler's batch size {}, too small for {} GPUs. Increasing to {}".format(
          sample_batch_size,
          self.pytorch.num_gpus,
          self.pytorch.num_gpus
        )
      )
      sample_batch_size = self.pytorch.num_gpus
    data_generator = torchLMDataGenerator.SampleMaskLMBatchGenerator(
                       self.config.training, sampler, self.tokenizer, self.config.training.random_seed, sample_batch_size,
                       self.config.architecture.max_position_embeddings, self.cache.path, corpus,
                       self.feature_encoder,
                       self.feature_tokenizer,
                       self.feature_sequence_length,
                     )
    self._ConfigSampleParams(data_generator, sampler)
    ckpt_step = self.loadCheckpoint(self.sample)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    if ckpt_step >= 0:
      l.logger().info("Loaded checkpoint step {}".format(ckpt_step))
    self.step_inputs   = None
    self.loader        = None
    self.pred_iterator = None
    l.logger().info("Initialized model samples in {}".format(self.sample_path / self.sampler.hash))
    return

  def InitSampleBatch(self, sampler: samplers.Sampler, **kwargs) -> None:
    """Batch-specific initialization. Called once when a new batch is going to be generated"""
    workload_size = kwargs.get('workload_size', None)
    if sampler.is_live:
      # For live sampling, start text must be re-instated at each iteration.
      self.sample = self.sample._replace(
        data_generator = torchLMDataGenerator.SampleMaskLMBatchGenerator(
          self.config.training, sampler, self.tokenizer, 0, sampler.batch_size,
          self.config.architecture.max_position_embeddings, self.cache.path,
          self.feature_encoder,
          self.feature_tokenizer,
          self.feature_sequence_length,
        )
      )
      self.step_inputs, self.loader, self.pred_iterator = None, None, None

    if self.loader is None:
      if self.torch_tpu_available:
        self.loader = self.pytorch.torch_ploader.ParallelLoader(
                          self.sample.data_generator.dataloader, [self.pytorch.device]
                    ).per_device_loader(self.pytorch.device)
      else:
        self.loader = self.sample.data_generator.dataloader

    if not sampler.is_active:
      if self.pred_iterator is None:
        self.pred_iterator = iter(self.loader)
      try:
        inputs = next(self.pred_iterator)
      except StopIteration:
        self.pred_iterator = iter(self.loader)
        inputs = next(self.pred_iterator)

      if workload_size is None:
        ## I think this dictionary holds tensors of the following size:
        ## [num_gpus x batch_size x seq_len] if only one node works.
        ## Otherwise, [1 x batch_size x seq_len] since each process manages its own GPU.
        padded_wsize = self.pytorch.num_gpus if environment.WORLD_SIZE == 1 else 1
      else:
        ## If a workload is specified, then after you pad to the dimension of GPU or num processes
        ## Divide the size by GPU size or num processes size.
        padded_wsize = (
          (max(1, workload_size // (self.pytorch.num_gpus * sampler.batch_size))) * self.pytorch.num_gpus
          if environment.WORLD_SIZE == 1
          else (workload_size // (self.pytorch.num_nodes * sampler.batch_size)) * self.pytorch.num_nodes)
      self.step_inputs = {
        x: inputs[x].unsqueeze(0).repeat(padded_wsize, 1, 1)
        for x in inputs
      }

      # This loop below is purely for proper printing reasons:
      sample_text = set(
        [self.tokenizer.tokensToString(
            seq.cpu().numpy(), ignore_token = self.tokenizer.padToken
          ) for seq in inputs['input_ids']]
      )
      for seq in sample_text:
        self.sampler.setStartText(seq)
        self.sampler.Specialize(self.tokenizer)
    return

  def SampleNextIndices(
    self, *unused_args, **unused_kwargs
  ) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    """Called iteratively to build a single batch of samples, until termination criteria stops calling"""
    del unused_kwargs
    del unused_args

    if self.sample is None:
      raise ValueError("Bert sampler has not been initialized.")

    with self.torch.no_grad():
      if self.sampler.is_active:
        try:
          return self.sample.data_generator.ActiveGeneration(self, self.sample)
        except StopIteration:
          raise StopIteration
      else:
        if self.sampler.is_live and self.feature_encoder:
          feat_space = ""
          while feat_space not in {"GreweFeatures", "AutophaseFeatures", "InstCountFeatures"}:
            feat_space = input("Select feature space: [g/a/i]/[GreweFeatures/AutophaseFeatures/InstCountFeatures]: ")
            if feat_space == "a":
              feat_space = "AutophaseFeatures"
            elif feat_space == "g":
              feat_space = "GreweFeatures"
            elif feat_space == "i":
              feat_space = "InstCountFeatures"
          input_features = {
            k: -1 for k in extractor.extractors.KEYS[feat_space]
          }
          for k in input_features.keys():
            if k not in {"F2:coalesced/mem", "F4:comp/mem"}:
              prompt = input("{}: ".format(k))
              if prompt == 0:
                val = 0
              else:
                val = int(prompt)
              input_features[k] = val
          if feat_space == "":
            feat_space = "GreweFeatures"
            input_features = {
              k: -1 for k in extractor.extractors.KEYS[feat_space]
            }
            for k in input_features.keys():
              if k not in {"F2:coalesced/mem", "F4:comp/mem"}:
                input_features[k] = int(np.random.poisson(8))
            print(input_features)
          if feat_space == "GreweFeatures":
            try:
              input_features["F2:coalesced/mem"] = input_features["coalesced"] / input_features["mem"]
            except ZeroDivisionError:
              input_features["F2:coalesced/mem"] = 0
            try:
              input_features["F4:comp/mem"] = input_features["comp"] / input_features["mem"]
            except ZeroDivisionError:
              input_features["F4:comp/mem"] = 0
          self.step_inputs['input_features'] = self.feature_tokenizer.TokenizeFeatureVector(input_features, feat_space, self.feature_sequence_length)
        l.logger().warn(self.step_inputs)
        step_out, time = self.sample_model_step(
            self.sample.model,
            self.step_inputs,
        )
        if self.pytorch.num_nodes > 1:
          self.torch.distributed.barrier()
          generated_samples = [self.torch.zeros(tuple(step_out['generated_samples'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
          sample_indices    = [self.torch.zeros(tuple(step_out['sample_indices'   ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
          self.torch.distributed.all_gather(generated_samples, step_out["generated_samples"])
          self.torch.distributed.all_gather(sample_indices,    step_out["sample_indices"])
          raise NotImplementedError("This will not work because generated_samples and sample indices are lists and not tensors")
        else:
          generated_samples = step_out['generated_samples']
          sample_indices    = step_out['sample_indices']

        return (
          self.step_inputs['original_input'].cpu().view(-1, self.step_inputs['original_input'].shape[2]).numpy(),
          self.step_inputs['input_ids'].cpu().view(-1, self.sampler.sequence_length).numpy(),
          generated_samples,
          sample_indices
        )

  def _getTestSampler(self, test_sampler, sequence_length):
    if test_sampler is None or test_sampler.is_live or test_sampler.is_active:
      if self.config.training.data_generator.HasField("hole"):
        sampler_str = [
            "start_text: \"[START]kernel void A([HOLE]}[END]\"",
            "batch_size: 2",
            "sequence_length: {}".format(sequence_length),
            "temperature_micros: 700000",
        ]
      elif self.config.training.data_generator.HasField("mask_seq"):
        sampler_str = [
            "start_text: \"[START]kernel void A(" + ''.join(["[MASK]"] * (sequence_length - 7)) + "}[END]\"",
            "batch_size: 2",
            "sequence_length: {}".format(sequence_length),
            "temperature_micros: 700000",
        ]
      mock_config = pbutil.FromString('\n'.join(sampler_str), sampler_pb2.Sampler())
      sampler = samplers.Sampler(mock_config, sample_db_name = "epoch_samples.db")
    else:
      sampler = test_sampler
    if sampler.isFixedStr:
      sampler.Specialize(self.tokenizer)
    observers = [sample_observers.PrintSampleObserver()]
    if FLAGS.store_samples_db:
      observers.append(sample_observers.SamplesDatabaseObserver(
          self.sample_path / sampler.hash / sampler.sample_db_name
        )
      )
      sampler.symlinkModelDB(
        self.sample_path / sampler.hash,
        self.hash
      )
    return sampler, observers

  def saveCheckpoint(self, estimator, pre_train):
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda x: self.ckpt_path / "{}{}-{}.pt".format("pre_" if pre_train else "", x, self.current_step)

      if self.torch_tpu_available:
        if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
          self.pytorch.torch_xla_model.save(estimator.model, ckpt_comp("model"))
        self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
        self.pytorch.torch_xla.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
        self.pytorch.torch_xla.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))
      else:
        if isinstance(estimator.model, self.torch.nn.DataParallel):
          self.torch.save(estimator.model.module.state_dict(), ckpt_comp("model"))
        else:
          self.torch.save(estimator.model.state_dict(), ckpt_comp("model"))
        self.torch.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
        self.torch.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))

      with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
        mf.write("{}train_step: {}\n".format("pre_" if pre_train else "", self.current_step))
      if pre_train:
        mf = open(self.ckpt_path / "checkpoint.meta", 'r')
        cf = mf.read()
        mf.close()
        if "train_step: 0" not in cf:
          with open(self.ckpt_path / "checkpoint.meta", 'w') as mf:
            mf.write(cf + "train_step: 0\n")
        for x in {"model"}:
          shutil.copyfile(str(ckpt_comp(x)), str(self.ckpt_path / "{}-0.pt".format(x)))
    return

  def loadCheckpoint(self,
                     estimator: typing.Union[
                                  typing.TypeVar('torchBert.BertEstimator'),
                                  typing.TypeVar('torchBert.SampleBertEstimator')
                                ],
                     pre_train          : bool = False,
                     without_label_head : bool = False,
                     is_decoder         : bool = False,
                     ) -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      if pre_train:
        key     = "pre_train_step"
        exclude = "None"
      else:
        key     = "train_step"
        exclude = "pre_train_step"
      get_step  = lambda x: int(x.replace("\n", "").replace("{}: ".format(key), ""))

      lines     = mf.readlines()
      entries   = set({get_step(x) for x in lines if key in x and exclude not in x})

    if FLAGS.select_checkpoint_step == -1 or pre_train:
      ckpt_step = max(entries)
    else:
      if FLAGS.select_checkpoint_step in entries:
        ckpt_step = FLAGS.select_checkpoint_step
      else:
        raise ValueError("{} not found in checkpoint folder.".format(FLAGS.select_checkpoint_step))

    ckpt_comp = lambda x: self.ckpt_path / "{}{}-{}.pt".format("pre_" if pre_train else "", x, ckpt_step)

    if isinstance(estimator.model, self.torch.nn.DataParallel):
      try:
        if without_label_head:
          new_state_dict = OrderedDict()
          for k, v in self.torch.load(ckpt_comp("model")).items():
            if "cls.predictions." not in k:
              new_state_dict[k] = v
          estimator.model.module.load_state_dict(new_state_dict, strict = False)
        else:
          estimator.model.module.load_state_dict(
            self.torch.load(ckpt_comp("model")),
            strict = False if is_decoder else True,
          )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          if not without_label_head or (without_label_head and "cls.predictions." not in name):
            new_state_dict[name] = v
        estimator.model.module.load_state_dict(new_state_dict, strict = False if is_decoder or without_label_head else True)
    else:
      try:
        if without_label_head:
          new_state_dict = OrderedDict()
          for k, v in self.torch.load(ckpt_comp("model")).items():
            if "cls.predictions." not in k:
              new_state_dict[k] = v
          estimator.model.load_state_dict(new_state_dict, strict = False)
        else:
          estimator.model.load_state_dict(
            self.torch.load(ckpt_comp("model")),
            strict = False if is_decoder else True,
          )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          if not without_label_head or (without_label_head and "cls.predictions." not in name):
            new_state_dict[name] = v
        estimator.model.load_state_dict(new_state_dict, strict = False if without_label_head or is_decoder else True)
    if isinstance(estimator, torchBert.BertEstimator):
      if estimator.optimizer is not None and estimator.scheduler is not None and ckpt_step > 0:
        estimator.optimizer.load_state_dict(
          self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device)
        )
        estimator.scheduler.load_state_dict(
          self.torch.load(ckpt_comp("scheduler"), map_location=self.pytorch.device)
        )
    estimator.model.eval()
    return ckpt_step

  def is_world_process_zero(self) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on
    several machines, this is only going to be :obj:`True` for one process).
    """
    if self.torch_tpu_available:
      return self.pytorch.torch_xla_model.is_master_ordinal(local=False)
    elif self.pytorch.num_nodes > 1:
      return self.torch.distributed.get_rank() == 0
    else:
      return True

  def count_parameters(self, model) -> int:
    """
    Count and print the number of trainable parameters for the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

  def find_unused_parameters(self, model: 'torch.nn.Module') -> None:
    """
    Find parameters that are unused for loss computation.
    """
    param_names = []
    for name, param in model.named_parameters():
      if param.grad is None:
        param_names.append(name)
    if param_names:
      l.logger().warn("Unused parameters:\n{}".format('\n'.join(param_names)))
    else:
      l.logger().info("No unused parameters found for grad computation.")
    return

  def GetShortSummary(self) -> str:

    return (
      "\n"
      f"{model_pb2.NetworkArchitecture.Backend.Name(self.config.architecture.backend)} "
      "network: "
      "\n"
      f" Total trainable parameters: {humanize.intcomma(self.count_parameters(self.train.model))}"
      "\n"
      f"  hidden_size: {self.config.architecture.hidden_size}"
      "\n"
      f"  #hidden_layers: {self.config.architecture.num_hidden_layers}"
      "\n"
      f"  #attention_heads: {self.config.architecture.num_attention_heads}"
      "\n"
      f"  intermediate_size: {self.config.architecture.intermediate_size}"
      "\n"
      f"  hidden_act: {self.config.architecture.hidden_act}"
      "\n"
    ) + (self.train.data_generator.GetShortSummary() if self.train else "")

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.
    Returns:
      A list of absolute paths.
    """
    # The TensorFlow save file.
    paths = [ path.absolute() for path in (self.cache.path / "checkpoints").iterdir() ]
    paths += [ path.absolute() for path in (self.cache.path / "logs").iterdir() ]
    paths += [ path.absolute() for path in (self.cache.path / "samples").iterdir() ]
    # paths += self.data_generator.InferenceManifest # TODO
    return sorted(paths)
