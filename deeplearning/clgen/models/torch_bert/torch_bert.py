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

from deeplearning.clgen import samplers
from deeplearning.clgen import sample_observers
from deeplearning.clgen import validation_database
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.models.torch_bert import model
from deeplearning.clgen.models.torch_bert import config
from deeplearning.clgen.models.torch_bert import optimizer
from deeplearning.clgen.models.torch_bert import hooks
from deeplearning.clgen.models.torch_bert.data_generator import MaskLMBatchGenerator

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "monitor_frequency",
  250,
  "Choose frequency (in steps) in which tensors will be logged during training. "
  "Default: 250"
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
    data_generator : MaskLMBatchGenerator
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
    self.telemetry           = None

    self.ckpt_path           = self._ConfigCheckpointParams()
    self.logfile_path        = self.cache.path / "logs"
    self.sample_path         = self.cache.path / "samples"

    self.is_validated        = False
    self.trained             = False
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
      self.bertAttrs, xla_device = self.torch_tpu_available
    )
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
    self.max_grad_norm                    = 1.0

    self.telemetry                        = telemetry.TrainingLogger(self.logfile_path)
    self.steps_per_epoch                  = data_generator.steps_per_epoch
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

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    # self.sample = tfBert.BertEstimator(tf.compat.v1.estimator.tpu.TPUEstimator(
    #                         use_tpu  = FLAGS.use_tpu,
    #                         model_fn = model_fn,
    #                         config   = run_config,
    #                         params   = {'sampling_temperature': sampler.temperature},
    #                         predict_batch_size = sampler.batch_size
    #                         ),
    #               data_generator
    #               )
    l.getLogger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  def samplesWithCategorical(self):
    return FLAGS.categorical_sampling

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
        MaskLMBatchGenerator.TrainMaskLMBatchGenerator(corpus, self.config.training, self.cache.path)
      )
    current_step = self.loadCheckpoint()
    l.getLogger().info("Loaded checkpoint step {}".format(current_step))
    self.is_trained = True if current_step >= self.num_train_steps else False

    if not self.is_trained:
      model = self.train.model.to(self.pytorch.device)
      model.zero_grad()

      if self.torch_tpu_available:
        total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
      else:
        dummy_num_machines = -1
        total_train_batch_size = (
          self.train_batch_size
          * (self.torch.distributed.get_world_size() if dummy_num_machines != -1 else 1)
        )

      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            self.train.data_generator.dataloader, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)
        self.train.data_generator.dataloader.sampler.set_epoch(current_step // self.steps_per_epoch)
      else:
        loader = self.train.data_generator.dataloader

      batch_iterator = iter(loader)    
      train_hook = hooks.tensorMonitorHook(
        self.logfile_path, current_step, min(self.steps_per_epoch, FLAGS.monitor_frequency)
      )

      l.getLogger().info(
        "Splitting {} steps into {} equivalent epochs, {} steps each. Rejected {} redundant step(s)".format(
          self.num_train_steps, self.num_epochs, 
          self.steps_per_epoch, self.config.training.num_train_steps - self.num_train_steps
        )
      )

      def training_step(model: typing.TypeVar('nn.Module'),
                        inputs: typing.Dict[str, typing.TypeVar('torch.Tensor')],
                        ) -> float:
        """
        Perform a training step on a batch of inputs.
        """
        for key, value in inputs.items():
          inputs[key] = value.to(self.pytorch.device)

        outputs = model(
                    input_ids           = inputs['input_ids'],
                    attention_mask      = inputs['input_mask'],
                    position_ids        = inputs['position_ids'],
                    labels              = inputs['mask_labels'],
                    next_sentence_label = inputs['next_sentence_label'],
                    masked_lm_lengths   = inputs['masked_lm_lengths'],
                  )
        total_loss, masked_lm_loss, next_sentence_loss = outputs[0], outputs[1], outputs[2]
        if self.pytorch.num_gpus > 1:
          total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
        total_loss.backward()
        return total_loss.item(), masked_lm_loss.item(), next_sentence_loss.item()

      try:
        model.train()
        for epoch in tqdm.auto.trange(self.num_epochs, desc="Epoch", leave = False):
          if epoch < current_step // self.steps_per_epoch:
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

            _, step_mask_loss, step_ns_loss = training_step(model, inputs)

            self.torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            if self.torch_tpu_available:
              self.pytorch.torch_xla.optimizer_step(self.train.optimizer)
            else:
              self.train.optimizer.step()
            self.train.scheduler.step()

            train_hook.step(
              masked_lm_loss = step_mask_loss,
              next_sentence_loss = step_ns_loss,
              total_loss = step_mask_loss + step_ns_loss,
              learning_rate = self.train.scheduler.get_last_lr()[0],
              execution_time_ms = int(round((datetime.datetime.utcnow() - start).total_seconds() * 1000))
            )

            model.zero_grad()
            current_step += 1

            # if self.args.evaluate_during_training and global_step % self.args.eval_steps == 0:
            #   self.evaluate()
          # End of Epoch
          self.saveCheckpoint(current_step)
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

    ###############
    model = self.train.model
    if self.pytorch.num_gpus > 1:
      model = self.torch.nn.DataParallel(model)

    eval_losses = []
    preds       = None
    label_ids   = None
    model.eval()

    if self.torch_tpu_available:
      loader = self.pytorch.torch_ploader.ParallelLoader(
                        self.train.data_generator.dataloader, [self.pytorch.device]
                  ).per_device_loader(self.pytorch.device)
    else:
      loader = self.train.data_generator.dataloader
    eval_iterator = iter(loader)

    def prediction_step(model: typing.TypeVar("torch.nn.Module"),
                        inputs: typing.Dict[str, typing.TypeVar("torch.Tensor")]
                        ):
      """
      Perform an evaluation step on :obj:`model` using obj:`inputs`.
      """
      for key, value in inputs.items():
        if isinstance(value, self.torch.Tensor):
          inputs[key] = value.to(self.pytorch.device)
      with self.torch.no_grad():
        outputs = model(
                    input_ids           = inputs['input_ids'],
                    attention_mask      = inputs['input_mask'],
                    position_ids        = inputs['position_ids'],
                    labels              = inputs['mask_labels'],
                    next_sentence_label = inputs['next_sentence_label'],
                    masked_lm_lengths   = inputs['masked_lm_lengths'],
                  )
        total_loss, masked_lm_loss, next_sentence_loss, logits = outputs[:4]
        total_loss = total_loss.mean().item()
      labels = inputs['mask_labels']
      return (total_loss, logits.detach(), labels.detach())

    for step in tqdm.auto.trange(FLAGS.max_eval_steps, desc = "Validation", leave = False):
      try:
        inputs = next(eval_iterator)
      except StopIteration:
        eval_iterator = iter(loader)
        inputs = next(eval_iterator)

      loss, preds, label_ids = prediction_step(model, inputs)
      batch_size = inputs[list(inputs.keys())[0]].shape[0]
      eval_losses.append(loss * batch_size)

    dummy_local_rank = -1
    if dummy_local_rank != -1:
      # In distributed mode, concatenate all results from all nodes:
      preds     = self.distributed_concat(
                      preds, num_total_examples=self.num_examples(loader)
                  )
      label_ids = self.distributed_concat(
                      label_ids, num_total_examples=self.num_examples(loader)
                  )
    elif self.torch_tpu_available:
      # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
      preds = self.pytorch.torch_xla_model.mesh_reduce("eval_preds", preds, self.torch.cat)
      label_ids = self.pytorch.torch_xla_model.mesh_reduce("eval_label_ids", label_ids, self.torch.cat)

    preds     = preds.cpu().numpy()
    label_ids = label_ids.cpu().numpy()

    # metrics   = self.compute_metrics(
    #               EvalPrediction(predictions=preds, label_ids=label_ids)
    #             )

    # metrics["eval_loss"] = np.sum(eval_losses) / (FLAGS.max_eval_steps * self.eval_batch_size)

    # Prefix all keys with eval_
    # for key in list(metrics.keys()):
    #   if not key.startswith("eval_"):
    #     metrics[f"eval_{key}"] = metrics.pop(key)

    # output = PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
    ###############

    # self.log(output.metrics)

    if self.pytorch.torch_tpu_available:
      # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
      self.pytorch.torch_xla_model.master_print(self.pytorch.torch_xla_met.metrics_report())

    self.is_validated = True
    return

  def loadCheckpoint(self):
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

    self.train.optimizer.load_state_dict(
      self.torch.load(ckpt_comp("optimizer"), map_location=self.pytorch.device)
    )
    self.train.scheduler.load_state_dict(
      self.torch.load(ckpt_comp("scheduler"))
    )
    # self.train.model = model.BertModel.from_pretrained(ckpt_comp("model"))
    self.train.model.load_state_dict(
      self.torch.load(ckpt_comp("model"))
    )
    self.train.model.eval()
    return ckpt_step

  def saveCheckpoint(self, step):
    """
      Saves model, scheduler, optimizer checkpoints per epoch.
    """
    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, step)

    if self.torch_tpu_available:
      if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
        self.pytorch.torch_xla_model.save(self.train.model, ckpt_comp("model"))
      self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
      self.pytorch.torch_xla.save(self.train.optimizer.state_dict(), ckpt_comp("optimizer"))
      self.pytorch.torch_xla.save(self.train.scheduler.state_dict(), ckpt_comp("scheduler"))
    elif self.is_world_process_zero():
      self.torch.save(self.train.model.state_dict(), ckpt_comp("model"))
      self.torch.save(self.train.optimizer.state_dict(), ckpt_comp("optimizer"))
      self.torch.save(self.train.scheduler.state_dict(), ckpt_comp("scheduler"))

    with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
      mf.write("train_step: {}\n".format(step))
    return

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

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    data_generator = MaskLMBatchGenerator.SampleMaskLMBatchGenerator(
                       sampler, self.atomizer, seed,
                       self.config.architecture.max_position_embeddings, self.cache.path
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
