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
import heapq
import math
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

  class Estimator(typing.NamedTuple):
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

    self.train  = None
    self.sample = None

    self.is_validated = False
    self.is_trained   = False

    self.eer_samples = eer_database.EERSamples(
      url        = "sqlite:///{}".format(str(self.sample_path / "samples.db")),
      must_exist = False,
    )
    self.sample_epoch = self.eer_samples.cur_sample_epoch
    l.logger().info("Active ExpectedErrorReduction config initialized in {}".format(self.cache_path))
    return

  def _ConfigModelParams(self) -> None:
    """
    Generic initialization.
    """
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
    return

  def _ConfigTrainParams(self, data_generator: 'torch.utils.data.Dataset') -> None:
    """
    Model parameter initialization.
    """
    if not self.train:
      self._ConfigModelParams()
      cm = model.MLP(self.model_config)
      if self.pytorch.num_nodes > 1:
        distrib.barrier()
        cm = self.torch.nn.parallel.DistributedDataParallel(
          cm,
          device_ids    = [self.pytorch.offset_device],
          output_device = self.pytorch.offset_device,
        )
      elif self.pytorch.num_gpus > 1:
        cm = self.torch.nn.DataParallel(cm)
      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = cm,
        num_train_steps = self.training_opts.num_train_steps,
        warmup_steps    = self.training_opts.num_warmup_steps,
        learning_rate   = self.training_opts.learning_rate,
      )
      self.train = ExpectedErrorReduction.Estimator(
          model          = cm,
          data_generator = data_generator,
          optimizer      = opt,
          scheduler      = lr_scheduler,
        )
      l.logger().info(self.GetShortSummary())
    return

  def _ConfigSampleParams(self) -> None:
    """
    Model parameter initialization.
    """
    if not self.sample:
      self._ConfigModelParams()
      cm = model.MLP(self.model_config)
      if self.pytorch.num_nodes > 1:
        distrib.barrier()
        cm = self.torch.nn.parallel.DistributedDataParallel(
          cm,
          device_ids    = [self.pytorch.offset_device],
          output_device = self.pytorch.offset_device,
        )
      elif self.pytorch.num_gpus > 1:
        cm = self.torch.nn.DataParallel(cm)
      self.sample = ExpectedErrorReduction.Estimator(
          model          = cm,
          data_generator = None,
          optimizer      = None,
          scheduler      = None,
        )
      l.logger().info(self.GetShortSummary())
    return

  def model_step(self,
                 model       : 'torch.nn.Module',
                 inputs      : typing.Dict[str, 'torch.Tensor'],
                 is_sampling : bool = False
                 ) -> float:
    """
    Run forward function for member model.
    """
    return model(
      input_ids   = inputs['input_ids'].to(self.pytorch.device),
      target_ids  = inputs['target_ids'].to(self.pytorch.device) if not is_sampling else None,
      is_sampling = is_sampling,
    )

  def Train(self, **kwargs) -> None:
    """
    Train the AL predictive model.
    """
    update_dataloader = kwargs.get('update_dataloader', None)
    if update_dataloader is None:
      l.logger().info("Initial EER model training.")
    self._ConfigTrainParams(self.downstream_task.data_generator)
    if not self.is_trained or update_dataloader is not None:

      data_generator  = (
        self.train.data_generator
        if update_dataloader is None
        else update_dataloader
             # + self.train.data_generator.get_random_subset(
                 # max(0, abs(len(update_dataloader) - self.training_opts.num_train_steps)))
      )
      if len(data_generator) == 0:
        return

      current_step = self.loadCheckpoint(self.train)
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
        self.train.model.zero_grad()

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
            self.train.model.train()
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
              step_out = self.model_step(self.train.model, inputs)
              # Backpropagate losses
              total_loss = step_out['total_loss'].mean()
              total_loss.backward()

              self.torch.nn.utils.clip_grad_norm_(self.train.model.parameters(), self.training_opts.max_grad_norm)
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
              self.train.model.zero_grad()
              if current_step == 0:
                l.logger().info("EER: Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))
              current_step += 1
            # End of epoch
            self.saveCheckpoint(self.train, step = current_step)
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
      raise NotImplementedError
    return

  def Sample(self, unlabelled_set: 'torch.Dataset') -> typing.List[typing.Dict[str, float]]:
    """
    Active learner sampling.

    unlabelled_set contains random datapoints provided by the downstream task.
    Expected Error Reduction algorithm is going to be applied for each datapoint for each label class.
    """

    l.logger().error("Problem #2: Check that for DDP, every one gets the chunk they must.")

    self._ConfigSampleParams()

    current_step = self.loadCheckpoint(self.sample)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    if current_step < 0:
      l.logger().warn("EER: You are trying to sample an untrained model.")
    current_step = max(0, current_step)

    # if self.pytorch.num_nodes <= 1:
    #   sampler = self.torch.utils.data.SequentialSampler(unlabelled_set)
    # else:
    #   sampler = self.torch.utils.data.DistributedSampler(
    #     unlabelled_set,
    #     num_replicas = self.pytorch.num_nodes,
    #     rank         = self.torch.distributed.get_rank(),
    #     shuffle      = False,
    #     drop_last    = False,
    #   )
    # loader = self.torch.utils.data.dataloader.DataLoader(
    #   dataset    = unlabelled_set,
    #   batch_size = self.training_opts.train_batch_size,
    #   sampler    = (sampler
    #     if self.pytorch.num_nodes <= 1 or not self.pytorch.torch_tpu_available or self.pytorch.torch_xla.xrt_world_size() <= 1
    #     else self.torch.utils.data.distributed.DistributedSampler(
    #       dataset      = unlabelled_set,
    #       num_replicas = self.pytorch.num_nodes if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.xrt_world_size(),
    #       rank         = self.torch.distributed.get_rank() if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.get_ordinal()
    #     )
    #   ),
    #   num_workers = 0,
    #   drop_last   = True if environment.WORLD_SIZE > 1 else False,
    # )
    # # Set dataloader in case of TPU training.
    # if self.torch_tpu_available:
    #   loader = self.pytorch.torch_ploader.ParallelLoader(
    #                       unlabelled_set, [self.pytorch.device]
    #                     ).per_device_loader(self.pytorch.device)
    # # Get dataloader iterator and setup hooks.
    # it = tqdm.tqdm(loader, desc="Sample Unlabelled set", leave = False) if self.is_world_process_zero() else loader

    ## If DDP, each node will work separately on chunks of the unlabelled dataset.
    node_size = len(unlabelled_set) // self.torch.distributed.get_world_size()
    node_set  = unlabelled_set[environment.WORLD_RANK * node_size: (1 + environment.WORLD_RANK) * node_size]
    if environment.WORLD_RANK == environment.WORLD_SIZE - 1:
      node_set += unlabelled_set[(1 + environment.WORLD_RANK) * node_size:]

    node_losses = {
      'input_ids'           : self.torch.zeros([len(node_set), self.downstream_task.input_size], dtype = self.torch.float32),
      'posterior_probs'     : self.torch.zeros([len(node_set), self.downstream_task.output_size], dtype = self.torch.float32),
      'aggregated_entropy'  : self.torch.zeros([len(node_set), self.downstream_task.output_size], dtype = self.torch.float32),
      'expected_error_rate' : self.torch.zeros([len(node_set), 1], dtype = self.torch.float32),
    }
    self.sample.model.eval()
    for idx, unl_train_point in enumerate(node_set):
      node_losses['input_ids'][idx] = unl_train_point['input_ids']
      for out_label in self.downstream_task.output_ids:

        ## For (x, y) run model inference to obtain p(x|y)
        out = self.model_step(self.sample.model, unl_train_point, is_sampling = True)
        node_losses['posterior_probs'][idx][out_label] = out['probs']

        ## Extend Dataset D+: D + (x, y)
        extended_dataset = self.downstream_task.dataset + {'input_ids': unl_train_point, 'target_ids': out_label}
        ## Copy the model to a temp one.
        new_model = copy.deepcopy(self.sample.model)

        ## Define optimizer, scheduler for training regime.
        opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
          model           = new_model,
          num_train_steps = len(extended_dataset),
          warmup_steps    = 0,
          learning_rate   = self.training_opts.learning_rate,
        )
        dp_estimator = ExpectedErrorReduction.Estimator(
          model          = new_model,
          data_generator = extended_dataset,
          optimizer      = opt,
          scheduler      = lr_scheduler,
        )
        ## Train the new model here.
        self.Train(eer_estimator = dp_estimator)
        ## Run the new model on the unlabelled dataset to estimate future errors.
        loader = self.torch.utils.data.dataloader.DataLoader(
          dataset     = node_set,
          batch_size  = self.training_opts.train_batch_size,
          sampler     = self.torch.utils.SequentialSampler(node_set),
          num_workers = 0,
          drop_last   = False,
        )
        aggr_entropy = 0.0
        target_ids = self.torch.zeros(
          [self.downstream_task, self.training_opts.train_batch_size], dtype = self.torch.int64
        )
        for tid in self.downstream_task.output_ids:
          target_ids[:,] = tid
        for unl_batch in iter(loader):
          for target_id_batch in target_ids:
            out = self.model_step(unl_batch['input_ids'], target_id_batch, is_sampling = False)
            aggr_entropy += out['loss']
        node_losses['aggregated_entropy'][idx][out_label] = aggr_entropy
      node_losses['expected_error_rate'][idx] = sum(
        [node_losses['posterior_probs'][L] * node_losses['aggregated_entropy'][L]
         for L in self.downstream_task.output_ids]
      )
    if self.pytorch.num_nodes > 1:
      self.torch.distributed.barrier()
      input_ids           = [self.torch.zeros(tuple(node_losses['input_ids'          ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      posterior_probs     = [self.torch.zeros(tuple(node_losses['posterior_probs'    ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      aggregated_entropy  = [self.torch.zeros(tuple(node_losses['aggregated_entropy' ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      expected_error_rate = [self.torch.zeros(tuple(node_losses['expected_error_rate'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]

      self.torch.distributed.all_gather(input_ids,           node_losses['input_ids'          ])
      self.torch.distributed.all_gather(posterior_probs,     node_losses['posterior_probs'    ])
      self.torch.distributed.all_gather(aggregated_entropy,  node_losses['aggregated_entropy' ])
      self.torch.distributed.all_gather(expected_error_rate, node_losses['expected_error_rate'])

      input_ids           = self.torch.reshape(input_ids,           (-1, input_ids.shape[-1]))
      posterior_probs     = self.torch.reshape(posterior_probs,     (-1, posterior_probs.shape[-1]))
      aggregated_entropy  = self.torch.reshape(aggregated_entropy,  (-1, aggregated_entropy.shape[-1]))
      expected_error_rate = self.torch.reshape(expected_error_rate, (-1, expected_error_rate.shape[-1]))

      expected_losses = {
        'input_ids'           : input_ids,
        'posterior_probs'     : posterior_probs,
        'aggregated_entropy'  : aggregated_entropy,
        'expected_error_rate' : expected_error_rate,
      }
    else:
      expected_losses = node_losses

    """
    for datapoint in unlabelled:
      avg_expected_loss['input_ids'] = datapoint['input_ids']

      for target in self.downstream_task.target_labels:
        out = model(datapoint['input_ids'])
        p(y|x) = out['probs'][target]

        new_dataset = self.downstream_task.labelled_dataset + (datappoint['target_ids'] = target)
        
        new_model = copy.deepcopy(model)
        new_model.Train(new_dataset)

        aggr_loss = 0.0
        for other_dp in unlabelled:
          for other_label in self.downstream_task.target_labels:
            out = new_model(other_dp['input_ids'])
            aggr_loss += LL[other_dp, other_label] = out['loss'][other_label]
        expected_losses[datapoint] = aggr_loss * p(y|x)

    to_be_labelled = keymin(expected_losses.values()) # Key with lowest value.
    """
    return

  def GetShortSummary(self) -> None:
    return "Short Summary TODO"