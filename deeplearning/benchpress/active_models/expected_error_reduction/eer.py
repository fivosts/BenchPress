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
A neural architecture for downstream task label prediction.

This head is used for feature-less learning to target benchmarks.
"""
import typing
import collections
import tqdm

from deeplearning.benchpress.models.torch_bert import hooks
from deeplearning.benchpress.active_models import backends
from deeplearning.benchpress.active_models import data_generator
from deeplearning.benchpress.active_models.expected_error_reduction import optimizer
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

    if environment.WORLD_RANK == 0:
      self.ckpt_path.mkdir(exist_ok = True, parents = True)
      self.sample_path.mkdir(exist_ok = True, parents = True)
      self.logfile_path.mkdir(exist_ok = True, parents = True)

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
      cm = model.MLP(self.model_config).to(self.pytorch.device)
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
        weight_decay    = 0.0,
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
      cm = model.MLP(self.model_config).to(self.pytorch.device)
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
    # The update dataloader for when you want to step-train after collecting target benchmark.
    update_dataloader = kwargs.get('update_dataloader', None)
    # Temp estimator, for when you are temp-training a model version during EER Sample.
    update_estimator  = kwargs.get('eer_estimator', None)

    if not update_estimator:
      # If not a temp estimator, then create the standard train estimator if not already created.
      self._ConfigTrainParams(self.downstream_task.data_generator)

    train_estimator = update_estimator if update_estimator else self.train

    if update_dataloader is None and update_estimator is None:
      l.logger().info("Initial EER model training.")
      # self.Validate()
    if not self.is_trained or update_dataloader is not None or update_estimator:

      data_generator  = (
        train_estimator.data_generator
        if update_dataloader is None
        else update_dataloader
             # + train_estimator.data_generator.get_random_subset(
                 # max(0, abs(len(update_dataloader) - self.training_opts.num_train_steps)))
      )
      if len(data_generator) == 0:
        return
      # ## TODO: Dummy code. If active learner can't learn on test set, then features suck.
      # Toggle this to train on test set. Used for evaluation purposes.
      # elif not update_estimator:
      #   data_generator = self.downstream_task.test_set

      # Empty cache for GPU environments.
      if self.pytorch.num_gpus > 0:
        self.torch.cuda.empty_cache()

      # Load most recent checkpoint to estimator, if not temp-model.
      if not update_estimator:
        current_step = self.loadCheckpoint(train_estimator)
        if current_step >= 0:
          l.logger().info("EER: Loaded checkpoint step {}".format(current_step))
        current_step = max(0, current_step)
        num_train_steps = min(
          (len(data_generator) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size,
          self.training_opts.num_train_steps
        ) if update_dataloader is None else ((len(update_dataloader) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size) + current_step
      else:
        current_step = 0
        num_train_steps = len(data_generator)

      if current_step < num_train_steps:
        train_estimator.model.zero_grad()

        # Setup sampler and data loader.
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
          drop_last   = False if environment.WORLD_SIZE == 1 else True,
        )
        # Set dataloader in case of TPU training.
        if self.torch_tpu_available:
          loader = self.pytorch.torch_ploader.ParallelLoader(
                              data_generator, [self.pytorch.device]
                            ).per_device_loader(self.pytorch.device)

        # Get dataloader iterator and setup hooks.
        batch_iterator = iter(loader)
        if self.is_world_process_zero() and not update_estimator:
          # Monitoring hook.
          train_hook = hooks.tensorMonitorHook(
            self.logfile_path,
            current_step,
            min(
              (len(data_generator) + self.training_opts.train_batch_size) // self.training_opts.train_batch_size,
              self.training_opts.steps_per_epoch, 50
            )
          )
        try:
          with self.torch.enable_grad():
            train_estimator.model.train()
            # epoch_iter = tqdm.auto.trange(self.training_opts.num_epochs, desc="Epoch", leave = False) if self.is_world_process_zero() else range(self.training_opts.num_epochs)
            # In distributed mode, calling the set_epoch() method at
            # the beginning of each epoch before creating the DataLoader iterator
            # is necessary to make shuffling work properly across multiple epochs.
            # Otherwise, the same ordering will be always used.

            if self.pytorch.num_nodes > 1:
              loader.sampler.set_epoch(current_step)

            batch_iter = tqdm.tqdm(batch_iterator, desc="Batch", leave = False) if self.is_world_process_zero() else batch_iterator
            for inputs in batch_iter:

              # Run model step on inputs
              step_out = self.model_step(train_estimator.model, inputs)
              # Backpropagate losses
              total_loss = step_out['total_loss'].mean()
              total_loss.backward()

              self.torch.nn.utils.clip_grad_norm_(train_estimator.model.parameters(), self.training_opts.max_grad_norm)
              if self.torch_tpu_available:
                self.pytorch.torch_xla.optimizer_step(train_estimator.optimizer)
              else:
                train_estimator.optimizer.step()
              train_estimator.scheduler.step()

              ## Collect tensors for logging.
              if self.pytorch.num_nodes > 1:
                total_loss = [
                  self.torch.zeros(tuple(step_out['total_loss'].shape), dtype = self.torch.float32).to(self.pytorch.device)
                  for _ in range(self.torch.distributed.get_world_size())
                ]
                self.torch.distributed.all_gather(total_loss, step_out["total_loss"])
              else:
                total_loss = step_out['total_loss'].unsqueeze(0).cpu()
              if self.is_world_process_zero() and not update_estimator:
                train_hook.step(
                  train_step = current_step,
                  total_loss = sum([tl.mean().item() for tl in total_loss]) / len(total_loss),
                )
              train_estimator.model.zero_grad()
              if current_step == 0 and update_estimator is None:
                l.logger().info("EER: Starting Loss: {}".format(sum([tl.mean().item() for tl in total_loss]) / len(total_loss)))
              current_step += 1
            # End of epoch
            if not update_estimator:
              self.saveCheckpoint(train_estimator, current_step = current_step)
            if self.is_world_process_zero() and not update_estimator:
              try:
                l.logger().info(
                  "EER: Step {} Loss: {}".format(
                    current_step, train_hook.epoch_loss
                  )
                )
              except ZeroDivisionError:
                l.logger().error(
                  "Hook has crashed again: current_step: {}, step_freq: {}, flush_freq: {}, train_step: {}".format(
                    train_hook.current_step, train_hook.step_freq, train_hook.flush_freq,
                    current_step
                  )
                )
              val_accuracy = self.Validate()
              train_hook.end_epoch(
                **{"val_{}_accuracy".format(key): val for key, val in val_accuracy.items()}
              )
            if self.torch_tpu_available:
              self.pytorch.torch_xla.master_print(self.pytorch.torch_xla_met.metrics_report())
        except KeyboardInterrupt:
          pass
    self.is_trained = True
    if self.pytorch.num_nodes > 1:
      self.torch.distributed.barrier()
    return

  def Validate(self, **kwargs) -> int:
    """
    Run validation to measure accuracy on the downstream task's selected test set, if exists.
    """
    # Load the test database from the downstream task.
    test_set = self.downstream_task.test_set
    # If non-empty.
    if test_set:
      _ = self.loadCheckpoint(self.train)
      self.train.model.zero_grad()

      # Setup sampler and dataloader.
      if self.pytorch.num_nodes <= 1:
        sampler = self.torch.utils.data.SequentialSampler(test_set)
      else:
        sampler = self.torch.utils.data.DistributedSampler(
          test_set,
          num_replicas = self.pytorch.num_nodes,
          rank         = self.pytorch.torch.distributed.get_rank()
        )
      loader = self.torch.utils.data.dataloader.DataLoader(
        dataset    = test_set,
        batch_size = self.training_opts.train_batch_size,
        sampler    = (sampler
          if not self.pytorch.torch_tpu_available or self.pytorch.torch_xla.xrt_world_size() <= 1
          else self.torch.utils.data.distributed.DistributedSampler(
            dataset      = test_set,
            num_replicas = self.pytorch.num_nodes if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.xrt_world_size(),
            rank         = self.pytorch.torch.distributed.get_rank() if not self.pytorch.torch_tpu_available else self.pytorch.torch_xla.get_ordinal()
          )
        ),
        num_workers = 0,
        drop_last   = False,
      )
      # Set dataloader in case of TPU training.
      if self.torch_tpu_available:
        loader = self.pytorch.torch_ploader.ParallelLoader(
                            data_generator, [self.pytorch.device]
                          ).per_device_loader(self.pytorch.device)
      # Setup iterator and accuracy metrics.
      batch_iter = tqdm.tqdm(iter(loader), desc = "Test Set", leave = False) if self.is_world_process_zero() else iter(loader)
      accuracy = {}
      missed_idxs = {}
      with self.torch.no_grad():
        self.train.model.eval()
        if self.pytorch.num_nodes > 1:
          loader.sampler.set_epoch(0)
        # Run inference.
        for inputs in batch_iter:

          step_out = self.model_step(self.train.model, inputs)

          ## Collect tensors for logging.
          if self.pytorch.num_nodes > 1:
            output_label = [
              self.torch.zeros(tuple(step_out['output_label'].shape), dtype = self.torch.int64).to(self.pytorch.device)
              for _ in range(self.torch.distributed.get_world_size())
            ]
            target_ids = [
              self.torch.zeros(tuple(inputs['target_ids'].shape), dtype = self.torch.int64).to(self.pytorch.device)
              for _ in range(self.torch.distributed.get_world_size())
            ]
            self.torch.distributed.all_gather(output_label, step_out['output_label'])
            self.torch.distributed.all_gather(target_ids, inputs['target_ids'].to(self.pytorch.device))
          else:
            output_label = step_out['output_label'].unsqueeze(0)
            target_ids   = inputs  ['target_ids'].unsqueeze(0).to(self.pytorch.device)

          # Group accuracy stats by label.
          # Assign to the first index the count of correct predictions.
          # Assign to the second index the total predictions.
          for id, label in zip(self.downstream_task.output_ids, self.downstream_task.output_labels):
            if label not in accuracy:
              accuracy[label] = [0, 0]
            accuracy[label][0] += int(self.torch.sum((output_label == id) & (target_ids == id)).cpu())
            accuracy[label][1] += int(self.torch.sum(target_ids == id).cpu())
          for out, tar, idx in zip(step_out['output_label'], inputs['target_ids'], inputs['idx']):
            if int(tar) != int(out):
              if int(tar) not in missed_idxs:
                missed_idxs[int(tar)] = []
              missed_idxs[int(tar)].append(int(idx))

      # You may want to all gather that.
      epoch_accuracy = {
        k: v[0] / v[1] for k, v in accuracy.items()
      }
      distrib.barrier()
    l.logger().error("Total data: {},\nValidation stats: {}\n{}".format(len(test_set), epoch_accuracy, accuracy))
    l.logger().error("Missed indices: {}".format(missed_idxs))
    return epoch_accuracy

  def Sample(self, sample_set: 'torch.Dataset') -> typing.List[typing.Dict[str, float]]:
    """
    Active learner sampling.

    sample_set contains random datapoints provided by the downstream task.
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

    ## If DDP, each node will work separately on chunks of the unlabelled dataset.
    node_size = len(sample_set) // environment.WORLD_SIZE
    node_rem  = len(sample_set) % environment.WORLD_SIZE
    node_set  = sample_set.get_sliced_subset(environment.WORLD_RANK * node_size, (1 + environment.WORLD_RANK) * node_size)
    if environment.WORLD_RANK == environment.WORLD_SIZE - 1 and node_rem > 0:
      node_set += sample_set.get_sliced_subset((1 + environment.WORLD_RANK) * node_size)

    node_loader = self.torch.utils.data.dataloader.DataLoader(
      dataset     = node_set,
      batch_size  = 1,
      sampler     = self.torch.utils.data.SequentialSampler(node_set),
      num_workers = 0,
      drop_last   = False,
    )

    node_losses = {
      'input_ids'           : self.torch.zeros([len(node_set), self.downstream_task.input_size],            dtype = self.torch.float32),
      'static_features'     : self.torch.zeros([len(node_set), self.downstream_task.static_features_size],  dtype = self.torch.float32),
      'runtime_features'    : self.torch.zeros([len(node_set), self.downstream_task.runtime_features_size], dtype = self.torch.int64),
      'posterior_probs'     : self.torch.zeros([len(node_set), self.downstream_task.output_size],           dtype = self.torch.float32),
      'aggregated_entropy'  : self.torch.zeros([len(node_set), self.downstream_task.output_size],           dtype = self.torch.float32),
      'expected_error_rate' : self.torch.zeros([len(node_set), 1],                                          dtype = self.torch.float32),
    }
    self.sample.model.eval()
    for idx, unl_train_point in tqdm.tqdm(enumerate(iter(node_loader)), total = len(node_loader), desc = "D + (x, y)"):
      node_losses['input_ids'][idx]        = unl_train_point['input_ids']
      node_losses['static_features'][idx]  = unl_train_point['static_features']
      node_losses['runtime_features'][idx] = unl_train_point['runtime_features']
      for out_label in self.downstream_task.output_ids:

        ## For (x, y) run model inference to obtain p(x|y)
        with self.torch.no_grad():
          out = self.model_step(self.sample.model, unl_train_point, is_sampling = True)
        node_losses['posterior_probs'][idx][out_label] = out['output_probs'].squeeze(0)[out_label]

        ## Extend Dataset D+: D + (x, y)
        # extended_dataset = self.downstream_task.dataset + {'input_ids': unl_train_point, 'target_ids': out_label}
        extended_datapoint = data_generator.ListTrainDataloader([], lazy = True)
        extended_datapoint.dataset = [
          {
            'input_ids':  unl_train_point['input_ids'].squeeze(0),
            'target_ids': self.torch.LongTensor([out_label]),
          }
        ]

        extended_dataset = self.downstream_task.data_generator + extended_datapoint
        ## Copy the model to a temp one.
        new_model = model.MLP(self.model_config).to(self.pytorch.device)
        if self.pytorch.num_nodes > 1:
          distrib.barrier()
          new_model.load_state_dict(self.sample.model.module.state_dict())
          new_model = self.torch.nn.parallel.DistributedDataParallel(
            new_model,
            device_ids    = [self.pytorch.offset_device],
            output_device = self.pytorch.offset_device,
          )
        elif self.pytorch.num_gpus > 1:
          new_model.load_state_dict(self.sample.model.module.state_dict())
          new_model = self.torch.nn.DataParallel(new_model)
        else:
          new_model.load_state_dict(self.sample.model.state_dict())

        ## Define optimizer, scheduler for training regime.
        opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
          model           = new_model,
          num_train_steps = len(extended_dataset),
          warmup_steps    = 0,
          learning_rate   = self.training_opts.learning_rate,
          weight_decay    = 0.0,
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
          sampler     = self.torch.utils.data.SequentialSampler(node_set),
          num_workers = 0,
          drop_last   = False,
        )
        aggr_entropy = 0.0
        target_ids = self.torch.zeros(
          [self.downstream_task.output_size, self.training_opts.train_batch_size, 1], dtype = self.torch.int64
        )
        with self.torch.no_grad():
          for tid in self.downstream_task.output_ids:
            target_ids[tid,:] = tid
          for unl_batch in iter(loader):
            for target_id_batch in target_ids:
              out = self.model_step(new_model, {'input_ids': unl_batch['input_ids'], 'target_ids': target_id_batch}, is_sampling = False)
              aggr_entropy += out['total_loss'].mean()
        node_losses['aggregated_entropy'][idx][out_label] = aggr_entropy

      node_losses['expected_error_rate'][idx] = sum(
        [node_losses['posterior_probs'][idx][L] * node_losses['aggregated_entropy'][idx][L]
         for L in self.downstream_task.output_ids]
      )
    if self.pytorch.num_nodes > 1:
      self.torch.distributed.barrier()
      input_ids           = [self.torch.zeros(tuple(node_losses['input_ids'          ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      static_features     = [self.torch.zeros(tuple(node_losses['static_features'    ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      runtime_features    = [self.torch.zeros(tuple(node_losses['runtime_features'   ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      posterior_probs     = [self.torch.zeros(tuple(node_losses['posterior_probs'    ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      aggregated_entropy  = [self.torch.zeros(tuple(node_losses['aggregated_entropy' ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      expected_error_rate = [self.torch.zeros(tuple(node_losses['expected_error_rate'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]

      self.torch.distributed.all_gather(input_ids,           node_losses['input_ids'          ])
      self.torch.distributed.all_gather(static_features,     node_losses['static_features'    ])
      self.torch.distributed.all_gather(runtime_features,    node_losses['runtime_features'   ])
      self.torch.distributed.all_gather(posterior_probs,     node_losses['posterior_probs'    ])
      self.torch.distributed.all_gather(aggregated_entropy,  node_losses['aggregated_entropy' ])
      self.torch.distributed.all_gather(expected_error_rate, node_losses['expected_error_rate'])

      input_ids           = self.torch.reshape(input_ids,           (-1, input_ids.shape[-1]))
      static_features     = self.torch.reshape(static_features,     (-1, static_features.shape[-1]))
      runtime_features    = self.torch.reshape(runtime_features,    (-1, runtime_features.shape[-1]))
      posterior_probs     = self.torch.reshape(posterior_probs,     (-1, posterior_probs.shape[-1]))
      aggregated_entropy  = self.torch.reshape(aggregated_entropy,  (-1, aggregated_entropy.shape[-1]))
      expected_error_rate = self.torch.reshape(expected_error_rate, (-1, expected_error_rate.shape[-1]))

      expected_losses = {
        'input_ids'           : input_ids,
        'static_features'     : static_features,
        'runtime_features'    : runtime_features,
        'posterior_probs'     : posterior_probs,
        'aggregated_entropy'  : aggregated_entropy,
        'expected_error_rate' : expected_error_rate,
      }
    else:
      expected_losses = node_losses

    expected_losses['input_ids']           = expected_losses['input_ids'          ].cpu().numpy()
    expected_losses['static_features']     = expected_losses['static_features'    ].cpu().numpy()
    expected_losses['runtime_features']    = expected_losses['runtime_features'   ].cpu().numpy()
    expected_losses['posterior_probs']     = expected_losses['posterior_probs'    ].cpu().numpy()
    expected_losses['aggregated_entropy']  = expected_losses['aggregated_entropy' ].cpu().numpy()
    expected_losses['expected_error_rate'] = expected_losses['expected_error_rate'].cpu().numpy()

    space_samples = []
    for idx in range(len(expected_losses['input_ids'])):
      space_samples.append({
        'input_ids'           : self.downstream_task.VecToInputFeatDict(expected_losses['input_ids'         ][idx]),
        'static_features'     : self.downstream_task.VecToStaticFeatDict(expected_losses['static_features'  ][idx]),
        'runtime_features'    : self.downstream_task.VecToRuntimeFeatDict(expected_losses['runtime_features'][idx]),
        'posterior_probs'     : expected_losses['posterior_probs'    ][idx],
        'aggregated_entropy'  : expected_losses['aggregated_entropy' ][idx],
        'expected_error_rate' : expected_losses['expected_error_rate'][idx],
      })
    return sorted(space_samples, key = lambda x: x['expected_error_rate'])

  def saveCheckpoint(self,
                     estimator    : 'ExpectedErrorReduction.Estimator',
                     current_step : int
                     ) -> None:
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, current_step)

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
        mf.write("train_step: {}\n".format(current_step))
    return

  def loadCheckpoint(self, estimator: 'ExpectedErrorReduction.Estimator') -> int:
    """
    Load model checkpoint. Loads either most recent epoch, or selected checkpoint through FLAGS.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1

    with open(self.ckpt_path / "checkpoint.meta", 'r') as mf:
      key     = "train_step"
      get_step  = lambda x: int(x.replace("\n", "").replace("{}: ".format(key), ""))

      lines     = mf.readlines()
      entries   = set({get_step(x) for x in lines if key in x})

    if FLAGS.select_checkpoint_step == -1:
      ckpt_step = max(entries)
    else:
      if FLAGS.select_checkpoint_step in entries:
        ckpt_step = FLAGS.select_checkpoint_step
      else:
        raise ValueError("{} not found in checkpoint folder.".format(FLAGS.select_checkpoint_step))

    ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, ckpt_step)

    if isinstance(estimator.model, self.torch.nn.DataParallel):
      try:
        estimator.model.module.load_state_dict(
          self.torch.load(ckpt_comp("model")),
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = collections.OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        estimator.model.module.load_state_dict(new_state_dict)
    else:
      try:
        estimator.model.load_state_dict(
          self.torch.load(ckpt_comp("model")),
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        new_state_dict = collections.OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        estimator.model.load_state_dict(new_state_dict)
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

  def GetShortSummary(self) -> None:
    return "Short Summary TODO"