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
from deeplearning.clgen.models.torch_bert.data_generator import MaskLMBatchGenerator

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

class torchBert(backends.BackendBase):

  class BertEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model      : typing.Any
    data_generator : MaskLMBatchGenerator

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
    l.getLogger().error("Issue with TORCH dropout probs: TORCH might not need inverse dropout prob: {}".format(
      self.config.architecture.hidden_dropout_prob
      )
    )
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

    self.telemetry                        = telemetry.TrainingLogger(self.logfile_path)
    self.steps_per_epoch                  = data_generator.steps_per_epoch
    self.num_epochs                       = data_generator.num_epochs
    self.num_train_steps                  = self.steps_per_epoch * self.num_epochs
    self.max_eval_steps                   = FLAGS.max_eval_steps

    self.validation_results_file          = "val_results.txt"
    self.validation_results_path          = os.path.join(str(self.logfile_path), self.validation_results_file)

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.train = torchBert.BertEstimator(
                    model.BertForPreTraining(self.bert_config).to(self.pytorch.device), 
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

  def training_step(self, 
                    model: typing.Any,# self.torch.nn.Module, 
                    inputs,#: typing.Dict[str, typing.Union[self.torch.Tensor, typing.Any]]
                    ) -> float:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
      model (:obj:`nn.Module`):
        The model to train.
      inputs (:obj:`Dict[str, Union[self.torch.Tensor, Any]]`):
        The inputs and targets of the model.

        The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
        argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
      :obj:`float`: The training loss on this batch.
    """
    model.train()
    for key, value in inputs.items():
      inputs[key] = value.to(self.pytorch.device)

    outputs = model(
                input_ids           = inputs['input_ids'],
                attention_mask      = inputs['input_mask'],
                position_ids        = inputs['position_ids'],
                labels              = inputs['mask_labels'],
                next_sentence_label = inputs['next_sentence_label']
              )
    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    loss = outputs[0]

    if self.counter % 50 == 0:
      l.getLogger().warn("Total loss: {}".format(loss))
    self.counter += 1

    if self.pytorch.num_gpus > 1:
      loss = loss.mean()  # mean() to average on multi-gpu parallel training

    loss.backward()
    return loss.item()

  def Train(self,
            corpus,
            test_sampler: typing.Optional[samplers.Sampler] = None,
            **unused_kwargs
            ) -> None:
    """
    Main training entry point.
    """
    self.counter = 0
    data_generator = MaskLMBatchGenerator.TrainMaskLMBatchGenerator(
                       corpus, self.config.training, self.cache.path
                     )

    self._ConfigTrainParams(data_generator)

    train_dataloader = self.train.data_generator.trainDataLoader()
    model = self.train.model.to(self.pytorch.device)

    opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
      model           = model,
      num_train_steps = self.num_train_steps,
      warmup_steps    = self.num_warmup_steps,
      learning_rate   = self.learning_rate,
    )

    """
    WARNING
    """
    # Check if saved optimizer or scheduler states exist
    # if (
    #   model_path is not None
    #   and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
    #   and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
    # ):
    #   # Load in optimizer and scheduler states
    #   opt.load_state_dict(
    #     self.torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
    #   )
    #   lr_scheduler.load_state_dict(self.torch.load(os.path.join(model_path, "scheduler.pt")))
    """
    WARNING
    """

    dummy_num_machines = -1
    dummy_gradient_accumulation_steps = 1
    dummy_max_grad_norm = 1.0
    dummy_logging_steps = 20

    # multi-gpu training (should be after apex fp16 initialization)
    if self.pytorch.num_gpus > 1:
      model = self.torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if dummy_num_machines != -1:
      model = self.torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dummy_num_machines],
        output_device=dummy_num_machines,
        find_unused_parameters=True,
      )

    # Train!
    if self.torch_tpu_available:
      total_train_batch_size = self.train_batch_size * self.pytorch.torch_xla.xrt_world_size()
    else:
      total_train_batch_size = (
        self.train_batch_size
        * (self.torch.distributed.get_world_size() if dummy_num_machines != -1 else 1)
      )
    l.getLogger().info("***** Running training *****")
    # logger.info("  Num examples = %d", self.num_examples(train_dataloader))
    l.getLogger().info("  Num Epochs = {}".format(self.num_epochs))
    # logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
    l.getLogger().info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(self.train_batch_size))
    l.getLogger().info("  Gradient Accumulation steps = {}".format(dummy_gradient_accumulation_steps))
    l.getLogger().info("  Total optimization steps = {}".format(self.num_train_steps))

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0



    """
    WARNING
    """
    # Check if continuing training from a checkpoint
    # if model_path is not None:
    #   # set global_step to global_step of last saved checkpoint from model path
    #   try:
    #     global_step = int(model_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // (len(train_dataloader) // dummy_gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (
    #       len(train_dataloader) // dummy_gradient_accumulation_steps
    #     )

    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    #   except ValueError:
    #     global_step = 0
    #     logger.info("  Starting fine-tuning.")
    """
    WARNING
    """


    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = tqdm.auto.trange(
      epochs_trained, int(np.ceil(self.num_epochs)), desc="Epoch")
    for epoch in train_iterator:

      if self.torch_tpu_available:
        parallel_loader = self.pytorch.torch_ploader.ParallelLoader(
                            train_dataloader, [self.args.device]
                          ).per_device_loader(self.args.device)
        train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm.auto.tqdm(parallel_loader, desc="Batch")
      else:
        epoch_iterator = tqdm.auto.tqdm(train_dataloader, desc="Batch")

      for step, inputs in enumerate(epoch_iterator):

        tr_loss += self.training_step(model, inputs)
        self.torch.nn.utils.clip_grad_norm_(model.parameters(), dummy_max_grad_norm)

        if self.torch_tpu_available:
          self.pytorch.torch_xla.optimizer_step(opt)
        else:
          opt.step()

        lr_scheduler.step()
        model.zero_grad()
        global_step += 1

        # if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (
        #   global_step == 1 and self.args.logging_first_step
        # ):
        if global_step == 1 or (global_step % 20 == 0):
          logs: Dict[str, float] = {}
          logs["loss"] = (tr_loss - logging_loss) / dummy_logging_steps
          # backward compatibility for pytorch schedulers
          logs["learning_rate"] = lr_scheduler.get_last_lr()[0]
          logging_loss = tr_loss
          # self.log(logs)

          # if self.args.evaluate_during_training and global_step % self.args.eval_steps == 0:
          #   self.evaluate()

          # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:

      if self.torch_tpu_available:
        self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())
    return

  def checkpointModel(self, global_step):
    # Save model checkpoint

    output_dir = lambda x: self.ckpt_path / "{}-{}.pt".format(x, global_step)
    self.save_model(output_dir)

    if self.is_world_process_zero():
      self._rotate_checkpoints()

    if self.torch_tpu_available:
      self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
      self.pytorch.torch_xla.save(self.train.optimizer.state_dict(), output_dir("optimizer"))
      self.pytorch.torch_xla.save(self.train.scheduler.state_dict(), output_dir("scheduler"))
    elif self.is_world_process_zero():
      self.torch.save(self.train.optimizer.state_dict(), output_dir("optimizer"))
      self.torch.save(self.train.scheduler.state_dict(), output_dir("scheduler"))

    return


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
