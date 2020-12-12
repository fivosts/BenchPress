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
import numpy as np
from absl import flags
import tqdm

from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import sample_observers
from deeplearning.clgen.samplers import validation_database
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.features import extractor
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.models.torch_bert import model
from deeplearning.clgen.models.torch_bert import config
from deeplearning.clgen.models.torch_bert import optimizer
from deeplearning.clgen.models.torch_bert import hooks
from deeplearning.clgen.models.torch_bert.data_generator import torchLMDataGenerator

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "monitor_frequency",
  250,
  "Choose frequency (in steps) in which tensors will be logged during training. "
  "Default: 250"
)

flags.DEFINE_boolean(
  "reward_compilation",
  False,
  "Select to integrate LLVM compiler into training regime."
  "During training, the target token will be asked to fill the first token of the hole."
  "If this flag is selected to True, the model will fill entirely the hole, as in inference."
  "The fully generated sample will be checked for syntactic correctness with LLVM."
  "If the sample compiles, then loss will be zero-ed for that instance, hence will be rewarded."
)

# flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

# flags.DEFINE_boolean("force_eval", False, "Run Validation no matter what.")

# flags.DEFINE_integer("sample_per_epoch", 3, "Set above zero to sample model after every epoch.")

# flags.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# flags.DEFINE_boolean("mirror_gpus", False, "Set True to distribute training across all system's GPUs. (Only usable when use_tpu is False).")

# flags.DEFINE_boolean("categorical_sampling", True, "Use categorical distribution on logits when sampling.")

class torchBert(backends.BackendBase):

  class BertEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : torchLMDataGenerator
    optimizer      : typing.Any
    scheduler      : typing.Any

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.bertAttrs           = None
    self.bert_config         = None

    self.train               = None
    self.sample              = None
    self.predict_generator   = None
    self.sampler             = None

    self.train_batch_size    = None
    self.eval_batch_size     = None
    self.learning_rate       = None
    self.num_train_steps     = None
    self.num_warmup_steps    = None

    self.ckpt_path           = self.cache.path / "checkpoints"
    self.logfile_path        = self.cache.path / "logs"
    self.sample_path         = self.cache.path / "samples"
    self.telemetry           = telemetry.TrainingLogger(self.logfile_path)

    self.is_validated        = False
    self.trained             = False
    l.getLogger().info("BERT Model config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, is_sampling):
    """General model hyperparameters initialization."""
    self.bertAttrs = {
          "vocab_size"                   : self.atomizer.vocab_size,
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
          "pad_token_id"                 : self.atomizer.padToken,
    }
    self.bert_config = config.BertConfig.from_dict(
      self.bertAttrs, xla_device = self.torch_tpu_available,
      reward_compilation = FLAGS.reward_compilation,
      is_sampling = is_sampling,
    )
    return

  def _ConfigTrainParams(self, 
                         data_generator: torchLMDataGenerator
                        ) -> None:
    """
    Model parameter initialization for training and validation.
    """
    self._ConfigModelParams(is_sampling = False)

    self.train_batch_size                 = self.config.training.batch_size
    self.eval_batch_size                  = self.config.training.batch_size
    self.learning_rate                    = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_warmup_steps                 = self.config.training.num_warmup_steps
    self.max_grad_norm                    = 1.0

    self.steps_per_epoch                  = data_generator.steps_per_epoch
    self.current_step                     = None
    self.num_epochs                       = data_generator.num_epochs
    self.num_train_steps                  = self.steps_per_epoch * self.num_epochs
    self.max_eval_steps                   = FLAGS.max_eval_steps

    self.validation_results_file          = "val_results.txt"
    self.validation_results_path          = os.path.join(str(self.logfile_path), self.validation_results_file)

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    m = model.BertForPreTraining(self.bert_config, atomizer = self.atomizer).to(self.pytorch.device)

    if self.pytorch.num_gpus > 1:
      m = self.torch.nn.DataParallel(m)

    dummy_num_machines = -1
    if dummy_num_machines != -1:
      m = self.torch.nn.parallel.DistributedDataParallel(
        m,
        device_ids=[dummy_num_machines],
        output_device=dummy_num_machines,
        find_unused_parameters=True,
      )

    opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
      model           = m,
      num_train_steps = self.num_train_steps,
      warmup_steps    = self.num_warmup_steps,
      learning_rate   = self.learning_rate,
    )

    self.train = torchBert.BertEstimator(
                  m, data_generator, opt, lr_scheduler
                )
    l.getLogger().info(self.GetShortSummary())
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

    m = model.BertForPreTraining(
        self.bert_config,
        atomizer = self.atomizer,
        use_categorical = FLAGS.categorical_sampling,
        temperature = self.temperature
      ).to(self.pytorch.device)
    if self.pytorch.num_gpus > 1:
      m = self.torch.nn.DataParallel(m)

    dummy_num_machines = -1
    if dummy_num_machines != -1:
      m = self.torch.nn.parallel.DistributedDataParallel(
        m,
        device_ids=[dummy_num_machines],
        output_device=dummy_num_machines,
        find_unused_parameters=True,
      )
    self.sample = torchBert.BertEstimator(
                    m, data_generator, None, None
                  )
    l.getLogger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  def samplesWithCategorical(self):
    return FLAGS.categorical_sampling

  def model_step(self,
                 model: typing.TypeVar('nn.Module'),
                 inputs: typing.Dict[str, typing.TypeVar('torch.Tensor')],
                 is_validation: bool = False,
                 ) -> float:
    """
    Perform a training step on a batch of inputs.
    """
    inputs['input_ids']            = inputs['input_ids'].to(self.pytorch.device)
    inputs['input_mask']           = inputs['input_mask'].to(self.pytorch.device)
    inputs['position_ids']         = inputs['position_ids'].to(self.pytorch.device)
    inputs['mask_labels']          = inputs['mask_labels'].to(self.pytorch.device)
    inputs['next_sentence_labels'] = inputs['next_sentence_labels'].to(self.pytorch.device)

    outputs = model(
                input_ids            = inputs['input_ids'],
                attention_mask       = inputs['input_mask'],
                position_ids         = inputs['position_ids'],
                masked_lm_labels     = inputs['mask_labels'],
                next_sentence_labels = inputs['next_sentence_labels'],
                is_validation        = is_validation,
              )
    return outputs

  def Train(self,
            corpus,
            test_sampler: typing.Optional[samplers.Sampler] = None,
            **unused_kwargs
            ) -> None:
    """
    Main training entry point.
    """
    if self.train is None:
      self._ConfigTrainParams(
        torchLMDataGenerator.TrainMaskLMBatchGenerator(corpus, self.config.training, self.cache.path)
      )

    if FLAGS.only_sample:
      return
      
    self.current_step = self.loadCheckpoint(self.train)
    if self.current_step > 0:
      l.getLogger().info("Loaded checkpoint step {}".format(self.current_step))
    self.is_trained = True if self.current_step >= self.num_train_steps else False

    if not self.is_trained:
      self.train.model.zero_grad()

      ## Set batch size in case of TPU training or distributed training.
      if self.torch_tpu_available:
        total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
      else:
        dummy_num_machines = -1
        total_train_batch_size = (
          self.train_batch_size
          * (self.torch.distributed.get_world_size() if dummy_num_machines != -1 else 1)
        )

      # Set dataloader in case of TPU training.
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            self.train.data_generator.dataloader, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)
        self.train.data_generator.dataloader.sampler.set_epoch(self.current_step // self.steps_per_epoch)
      else:
        loader = self.train.data_generator.dataloader

      # Get dataloader iterator and setup hooks.
      batch_iterator = iter(loader)    
      train_hook = hooks.tensorMonitorHook(
        self.logfile_path, self.current_step, min(self.steps_per_epoch, FLAGS.monitor_frequency)
      )
      if FLAGS.reward_compilation:
        correct_sample_obs = sample_observers.SamplesDatabaseObserver(
          self.logfile_path / "correct_samples.db"
        )
      else:
        correct_sample_obs = None
        
      l.getLogger().info(
        "Splitting {} steps into {} equivalent epochs, {} steps each. Rejected {} redundant step(s)".format(
          self.num_train_steps, self.num_epochs, 
          self.steps_per_epoch, self.config.training.num_train_steps - self.num_train_steps
        )
      )

      try:
        self.train.model.train()
        for epoch in tqdm.auto.trange(self.num_epochs, desc="Epoch", leave = False):
          if epoch < self.current_step // self.steps_per_epoch:
            continue # Stupid bar won't resume.
          
          for step in tqdm.auto.trange(self.steps_per_epoch, desc="Batch", leave = False):
            start = datetime.datetime.utcnow()
            try:
              inputs = next(batch_iterator)
            except StopIteration:
              # dataloader has different len() than steps_per_epoch.
              # This is the easiest way to infinite-loop dataloaders in pytorch.
              batch_iterator = iter(loader)
              inputs = next(batch_iterator)

            step_out = self.model_step(self.train.model, inputs)
            total_loss = step_out['total_loss'].mean()
            total_loss.backward()

            self.torch.nn.utils.clip_grad_norm_(self.train.model.parameters(), self.max_grad_norm)
            if self.torch_tpu_available:
              self.pytorch.torch_xla.optimizer_step(self.train.optimizer)
            else:
              self.train.optimizer.step()
            self.train.scheduler.step()

            exec_time_ms = int(round((datetime.datetime.utcnow() - start).total_seconds() * 1000))
            if FLAGS.reward_compilation:
              correct_samples = [(x, y) for en, (x, y) in enumerate(zip(inputs['input_ids'].cpu().numpy(), step_out['generated_samples'].cpu().numpy())) if step_out['compile_status'][en] == 1]
              for s in correct_samples:
                feature_vector = extractor.DictKernelFeatures(self.atomizer.ArrayToCode(s[1]))
                correct_sample_obs.OnSample(model_pb2.Sample(
                    train_step             = self.current_step,
                    sample_feed            = self.atomizer.DeatomizeIndices(s[0], ignore_token = self.atomizer.padToken).replace("\\n", "\n"),
                    text                   = self.atomizer.DeatomizeIndices(s[1], ignore_token = self.atomizer.padToken).replace("\\n", "\n"),
                    encoded_text           = ",".join([str(t) for t in s[1]]),
                    sample_indices         = '',
                    encoded_sample_indices = '',
                    sample_time_ms         = int(round(exec_time_ms / self.train_batch_size)),
                    feature_vector         = "\n".join(["{}:{}".format(k, v) for (k, v) in feature_vector.items()]),
                    num_tokens             = len([x for x in s[1] if x != self.atomizer.padToken]),
                    categorical_sampling   = False,
                    compile_status         = True,
                    date_added             = datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"),
                  )
                )
            train_hook.step(
              masked_lm_loss          = step_out['masked_lm_loss'].mean().item(),
              next_sentence_loss      = step_out['next_sentence_loss'].mean().item(),
              total_loss              = total_loss.item(),
              learning_rate           = self.train.scheduler.get_last_lr()[0],
              compilation_rate        = step_out['batch_compilation_rate'].mean().item(),
              batch_execution_time_ms = exec_time_ms,
              time_per_sample_ms      = exec_time_ms / self.train_batch_size,
              num_correct_samples     = (correct_sample_obs.sample_id if correct_sample_obs is not None else None)
            )
            self.train.model.zero_grad()
            if self.current_step == 0:
              l.getLogger().info("Starting Loss: {}".format(total_loss.item()))
            self.current_step += 1

          # End of Epoch
          l.getLogger().info("Epoch {} Loss: {}".format(self.current_step // self.steps_per_epoch, train_hook.current_loss))
          self.saveCheckpoint(self.train)
          if self.torch_tpu_available:
            self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())

        self.is_trained = True
      except KeyboardInterrupt:
        pass

      if not FLAGS.force_eval:
        self.Validate()

    if FLAGS.force_eval and not self.is_validated:
      self.Validate()
    return

  def Validate(self) -> None:

    if self.pytorch.num_gpus > 1:
      model = self.torch.nn.DataParallel(self.train.model)

    avg_mask_loss = []
    avg_nsp_loss  = []
    preds       = None
    label_ids   = None
    self.train.model.eval()

    for set_name, dataloader in self.train.data_generator.eval_dataloaders():
      l.getLogger().info("BERT Validation on {}".format(set_name))
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                          self.train.data_generator.dataloader, [self.pytorch.device]
                    ).per_device_loader(self.pytorch.device)
      else:
        loader = self.train.data_generator.dataloader

      val_hook = hooks.validationSampleHook(
        url = "sqlite:///{}".format(str(self.logfile_path / "validation_samples.db")),
        atomizer = self.atomizer,
        batch_size = self.eval_batch_size,
        model_step = self.current_step
      )
      eval_iterator = iter(loader)

      for step in tqdm.auto.trange(FLAGS.max_eval_steps, desc = "Validation", leave = False):
        try:
          inputs = next(eval_iterator)
        except StopIteration:
          eval_iterator = iter(loader)
          inputs = next(eval_iterator)

        with self.torch.no_grad():
          step_out = self.model_step(self.train.model, inputs, is_validation = True)

        val_hook.step(inputs, step_out)
        avg_mask_loss.append(step_out['masked_lm_loss'].mean().item())
        avg_nsp_loss.append(step_out['next_sentence_loss'].mean().item())

      val_hook.final(set_name, avg_mask_loss, avg_nsp_loss)
      if self.pytorch.torch_tpu_available:
        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        self.pytorch.torch_xla_model.master_print(self.pytorch.torch_xla_met.metrics_report())

    self.is_validated = True
    return

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    data_generator = torchLMDataGenerator.SampleMaskLMBatchGenerator(
                       self.config.training, sampler, self.atomizer, seed,
                       self.config.architecture.max_position_embeddings, self.cache.path
                     )
    self._ConfigSampleParams(data_generator, sampler)
    ckpt_step = self.loadCheckpoint(self.sample)
    self.step_inputs   = None
    self.loader        = None
    self.pred_iterator = None
    l.getLogger().info("Initialized model samples in {}".format(self.sample_path))
    return

  def InitSampleBatch(self, *unused_args, **unused_kwargs) -> None:
    """Batch-specific initialization. Called once when a new batch is going to be generated"""
    del unused_args
    del unused_kwargs

    if self.loader is None:
      if self.torch_tpu_available:
        self.loader = self.pytorch.torch_ploader.ParallelLoader(
                          self.sample.data_generator.dataloader, [self.pytorch.device]
                    ).per_device_loader(self.pytorch.device)
      else:
        self.loader = self.sample.data_generator.dataloader

    if self.pred_iterator is None:
      self.pred_iterator = iter(self.loader)
  
    try:
      inputs = next(self.pred_iterator)
    except StopIteration:
      self.pred_iterator = iter(self.loader)
      inputs = next(self.pred_iterator)

    self.step_inputs = {x: inputs[x].repeat((self.sampler.batch_size, 1)) for x in inputs}
    self.sampler.setStartText(self.atomizer.DeatomizeIndices(inputs['input_ids'][0].cpu().numpy(), ignore_token = self.atomizer.padToken))
    self.sampler.Specialize(self.atomizer)
    return

  def SampleNextIndices(self, *unused_args, **unused_kwargs):
    """Called iteratively to build a single batch of samples, until termination criteria stops calling"""
    del unused_kwargs
    del unused_args
    if self.sample is None:
      raise ValueError("Bert sampler has not been initialized.")
    # if self.pytorch.num_gpus > 1:
    #   model = self.torch.nn.DataParallel(self.sample.model)
    self.sample.model.eval()
    with self.torch.no_grad():
      step_out = self.model_step(
          self.sample.model, self.step_inputs,
      )
      if self.sampler.is_active:
        generated_samples, sample_indices = step_out['generated_samples'].cpu().numpy(), step_out['sample_indices'].cpu().numpy()
        while True:
          active_sample, active_indices, done = self.sample.data_generator.EvaluateFeatures(
            np.asarray(generated_samples),
            np.asarray(sample_indices)
          )
          if done:
            return active_sample, active_indices
          else:
            step_input = {
              x: active_sample[x].repeat((self.sampler.batch_size, 1)) for x in active_sample
            }
            self.sampler.setStartText(
              self.atomizer.DeatomizeIndices(
                step_input['input_ids'][0].cpu().numpy(), ignore_token = self.atomizer.padToken
              )
            )
            self.sampler.Specialize(self.atomizer)
            active_step = self.model_step(
                self.sample.model, step_input,
            )
            generated_samples, sample_indices = active_step.generated_samples, active_step.sample_indices
      else:
        return step_out['generated_samples'].cpu().numpy(), step_out['sample_indices'].cpu().numpy()
    raise ValueError("While True loop broken without returning")

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
          self.sample_path / sampler.hash / sampler.sample_db_name
        )
      )
      sampler.symlinkModelDB(
        self.sample_path / sampler.hash,
        self.hash
      )
    return sampler, observers

  def saveCheckpoint(self, estimator):
    """
      Saves model, scheduler, optimizer checkpoints per epoch.
    """
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, self.current_step)

    if self.torch_tpu_available:
      if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
        self.pytorch.torch_xla_model.save(estimator.model, ckpt_comp("model"))
      self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
      self.pytorch.torch_xla.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
      self.pytorch.torch_xla.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))
    elif self.is_world_process_zero():
      self.torch.save(estimator.model.state_dict(), ckpt_comp("model"))
      self.torch.save(estimator.optimizer.state_dict(), ckpt_comp("optimizer"))
      self.torch.save(estimator.scheduler.state_dict(), ckpt_comp("scheduler"))

    with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
      mf.write("train_step: {}\n".format(self.current_step))
    return

  def loadCheckpoint(self, estimator):
    """
      Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return 0

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      get_step  = lambda x: int(x.replace("\n", "").replace("train_step: ", ""))
      entries   = set({get_step(x) for x in mf.readlines()})

    if FLAGS.select_checkpoint_step == -1:
      ckpt_step = max(entries)
    else:
      if FLAGS.select_checkpoint_step in entries:
        ckpt_step = FLAGS.select_checkpoint_step
      else:
        raise ValueError("{} not found in checkpoint folder.".format(FLAGS.select_checkpoint_step))

    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, ckpt_step)

    # self.train.model = model.BertModel.from_pretrained(ckpt_comp("model"))
    estimator.model.load_state_dict(
      self.torch.load(ckpt_comp("model"))
    )
    if estimator.optimizer is not None and estimator.scheduler is not None:
      estimator.optimizer.load_state_dict(
        self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device)
      )
      estimator.scheduler.load_state_dict(
        self.torch.load(ckpt_comp("scheduler"))
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
    else:
      # TODO
      dummy_local_rank = -1
      return dummy_local_rank == -1 or self.torch.distributed.get_rank() == 0

  def GetShortSummary(self) -> str:
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
