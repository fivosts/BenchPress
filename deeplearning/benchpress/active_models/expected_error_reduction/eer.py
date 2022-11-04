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
"""
A neural architecture for CPU vs GPU device mapping prediction.

This head is used for feature-less learning to target benchmarks.
"""
import typing
import pathlib

from deeplearning.benchpress.models.torch_bert import optimizer
from deeplearning.benchpress.models.torch_bert import hooks
from deeplearning.benchpress.active_models import backends
from deeplearning.benchpress.active_models import data_generator
from deeplearning.benchpress.active_models.expected_error_reduction import model
from deeplearning.benchpress.active_models.expected_error_reduction import config
from deeplearning.benchpress.active_models.expected_error_reduction import eer_database
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib

from deeplearning.benchpress.util import logging as l

from absl import flags

FLAGS = flags.FLAGS

class ExpectedErrorReduction(backends.BackendBase):

  class TrainingOpts(typing.NamedTuple):
    """Wrapper class for training options"""
    train_batch_size : int
    learning_rate    : float
    num_warmup_steps : int
    max_grad_norm    : float
    steps_per_epoch  : int
    num_epochs       : int
    num_train_steps  : int

  class CommitteeEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : 'torch.utils.data.Dataset'
    optimizer      : typing.Any
    scheduler      : typing.Any

  def __repr__(self):
    return "ExpectedErrorReduction"

  def __init__(self, *args, **kwargs):

    super(ExpectedErrorReduction, self).__init__(*args, **kwargs)
    from deeplearning.benchpress.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.random_seed)
    self.torch.cuda.manual_seed_all(self.config.random_seed)

    self.ckpt_path    = self.cache_path / "checkpoints"
    self.sample_path  = self.cache_path / "samples"
    self.logfile_path = self.cache_path / "logs"

    self.validation_results_file = "val_results.txt"
    self.validation_results_path = self.logfile_path / self.validation_results_file

    self.model_config  = None
    self.training_opts = None
    self.estimator     = None

    self.is_validated = False
    self.is_trained   = False

    self.eer_samples = eer_database.EERSamples(
      url        = "sqlite:///{}".format(str(self.sample_path / "samples.db")),
      must_exist = False,
    )
    self.sample_epoch = self.eer_samples.cur_sample_epoch
    l.logger().info("Active ExpectedErrorReduction config initialized in {}".format(self.cache_path))
    return

  def _ConfigModelParams(self, data_generator: 'torch.utils.data.Dataset' = None) -> None:
    """
    Model parameter initialization.
    """
    if not self.estimator:
      self.model_config = config.ModelConfig.FromConfig(
        self.config.expected_error_reduction, 
        self.downstream_task,
        self.config.num_train_steps
      )
      self.training_opts = ExpectedErrorReduction.TrainingOpts(
        train_batch_size = self.model_config.batch_size,
        learning_rate    = self.model_config.learning_rate,
        num_warmup_steps = self.model_config.num_warmup_steps,
        max_grad_norm    = self.model_config.max_grad_norm,
        steps_per_epoch  = self.model_config.steps_per_epoch,
        num_epochs       = self.model_config.num_epochs,
        num_train_steps  = self.model_config.num_train_steps,
      )
      cm = model.MLP(self.model_config.head)
      if self.pytorch.num_nodes > 1:
        distrib.barrier()
        cm = self.torch.nn.parallel.DistributedDataParallel(
          cm,
          device_ids             = [self.pytorch.offset_device],
          output_device          = self.pytorch.offset_device,
        )
      elif self.pytorch.num_gpus > 1:
        cm = self.torch.nn.DataParallel(cm)
      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = cm,
        num_train_steps = self.training_opts.num_train_steps,
        warmup_steps    = self.training_opts.num_warmup_steps,
        learning_rate   = self.training_opts.learning_rate,
      )
      self.estimator = ExpectedErrorReduction.Estimator(
          model          = cm,
          data_generator = data_generator,
          optimizer      = opt,
          scheduler      = lr_scheduler,
        )
      (self.ckpt_path / self.model_config.sha256).mkdir(exist_ok = True, parents = True),
      (self.logfile_path / self.model_config.sha256).mkdir(exist_ok = True, parents = True),
      l.logger().info(self.GetShortSummary())
    return

  def model_step(self, inputs: typing.Dict[str, 'torch.Tensor'], is_sampling: bool = False) -> float:
    """
    Run forward function for member model.
    """
    outputs = self.estimator.model(
      input_ids   = inputs['input_ids'].to(self.pytorch.device),
      target_ids  = inputs['target_ids'].to(self.pytorch.device) if not is_sampling else None,
      is_sampling = is_sampling,
    )
    return outputs

  def Train(self, **kwargs) -> None:
    """
    Train the AL predictive model.
    """
    update_dataloader = kwargs.get('update_dataloader', None)
    if update_dataloader is None:
      l.logger().info("Initial EER model training.")
    self._ConfigModelParams(self.downstream_task.data_generator)
    if not self.is_trained or update_dataloader is not None:

      data_generator  = (
        self.estimator.data_generator
        if update_dataloader is None
        else update_dataloader
             # + self.estimator.data_generator.get_random_subset(
                 # max(0, abs(len(update_dataloader) - self.training_opts.num_train_steps)))
      )
      if len(data_generator) == 0:
        return

      current_step = self.loadCheckpoint(self.estimator)
      if self.pytorch.num_gpus > 0:
        self.torch.cuda.empty_cache()
      if current_step >= 0:
        l.logger().info("EER: Loaded checkpoint step {}".format(current_step))
      current_step = max(0, current_step)
      num_train_steps = min(
        (len(data_generator) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size,
        self.training_opts.num_train_steps
      ) if update_dataloader is None else ((len(update_dataloader) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size) + current_step

      if current_step < num_train_steps:
        self.estimator.model.zero_grad()

        if self.pytorch.num_nodes <= 1:
          sampler = self.torch.utils.data.RandomSampler(data_generator, replacement = False)
        else:
          sampler = self.torch.utils.data.DistributedSampler(
            data_generator,
            num_replicas = self.pytorch.num_nodes,
            rank         = self.pytorch.torch.distributed.get_rank()
          )
        loader = self.torch.utils.data.dataloader.DataLoader(
          dataset    = data_generator,
          batch_size = self.training_opts.train_batch_size,
          sampler    = (sampler
            if not self.pytorch.torch_tpu_available or self.pytorch.torch_xla.xrt_world_size() <= 1
            else self.torch.utils.data.distributed.DistributedSampler(
              dataset      = data_generator,
              num_replicas = self.pytorch.num_nodes if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.xrt_world_size(),
              rank         = self.pytorch.torch.distributed.get_rank() if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.get_ordinal()
            )
          ),
          num_workers = 0,
          drop_last   = False if self.torch.distributed.get_world_size() == 1 else True,
        )
        # Set dataloader in case of TPU training.
        if self.torch_tpu_available:
          loader = self.pytorch.torch_ploader.ParallelLoader(
                              data_generator, [self.pytorch.device]
                            ).per_device_loader(self.pytorch.device)

        # Get dataloader iterator and setup hooks.
        batch_iterator = iter(loader)
        if self.is_world_process_zero():
          train_hook = hooks.tensorMonitorHook(
            member_log_path,
            current_step,
            min(
              (len(data_generator) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size,
              self.training_opts.steps_per_epoch, 50
            )
          )
        try:
          with self.torch.enable_grad():
            self.estimator.model.train()
            # epoch_iter = tqdm.auto.trange(self.training_opts.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(self.training_opts.num_epochs)
            epoch = num_train_steps // self.training_opts.steps_per_epoch
            # In distributed mode, calling the set_epoch() method at
            # the beginning of each epoch before creating the DataLoader iterator
            # is necessary to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.

            if self.pytorch.num_nodes > 1:
              loader.sampler.set_epoch(epoch)

            batch_iter = tqdm.tqdm(batch_iterator, desc="Batch", leave = False) if self.is_world_process_zero() else batch_iterator
            for inputs in batch_iter:
              if self.is_world_process_zero():
                start = datetime.datetime.utcnow()

              # Run model step on inputs
              step_out = self.model_step(inputs)
              # Backpropagate losses
              total_loss = step_out['total_loss'].mean()
              total_loss.backward()

              self.torch.nn.utils.clip_grad_norm_(self.estimator.model.parameters(), self.training_opts.max_grad_norm)
              if self.torch_tpu_available:
                self.pytorch.torch_xla.optimizer_step(optimizer)
              else:
                optimizer.step()
              scheduler.step()

              ## Collect tensors for logging.
              if self.pytorch.num_nodes > 1:
                total_loss = [
                  self.torch.zeros(tuple(step_out['total_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device)
                  for _ in range(self.torch.distributed.get_world_size())
                ]
                self.torch.distributed.all_gather(total_loss, step_out["total_loss"])
              else:
                total_loss = step_out['total_loss'].unsqueeze(0).cpu()
              if self.is_world_process_zero():
                train_hook.step(
                  train_step = current_step,
                  total_loss = sum([tl.mean().item() for tl in total_loss]) / len(total_loss),
                )
              self.estimator.model.zero_grad()
              if current_step == 0:
                l.logger().info("EER: Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))
              current_step += 1
            # End of epoch
            self.saveCheckpoint(self.estimator, member_path,step = current_step)
            if self.is_world_process_zero():
              try:
                l.logger().info(
                  "EER: Epoch {} Loss: {}".format(
                    current_step // self.training_opts.steps_per_epoch, train_hook.epoch_loss
                  )
                )
              except ZeroDivisionError:
                l.logger().error(
                  "Hook has crashed again: current_step: {}, step_freq: {}, flush_freq: {}, train_step: {}".format(
                    train_hook.current_step, train_hook.step_freq, train_hook.flush_freq,
                    current_step
                  )
                )
              train_hook.end_epoch()
            if self.torch_tpu_available:
              self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())
        except KeyboardInterrupt:
          pass
    self.is_trained = True
    if self.pytorch.num_nodes > 1:
      self.torch.distributed.barrier()
    return

  def Validate(self, **kwargs):
    """
    Run validation to measure accuracy on the downstream task's selected test set, if exists.
    """
    test_set = self.downstream_task.test_set
    if test_set:
      #1. load checkpoint.
      #2. run model on test set
      #3. get accuracy metrics.
      pass
    return

  def Sample(self, sample_set: 'torch.Dataset') -> typing.List[typing.Dict[str, float]]:
    """
    Active learner sampling.

    sample_set contains random datapoints provided by the downstream task.
    Expected Error Reduction algorithm is going to be applied for each datapoint for each label class.
    """
    raise NotImplementedError
    return