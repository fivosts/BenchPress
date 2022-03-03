"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import pathlib
import copy

from deeplearning.clgen.models import backends
from deeplearning.clgen.util.pytorch import torch

class ActiveCommittee(backends.BackendBase):

  class CommitteeEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : torchLMDataGenerator
    optimizer      : typing.Any
    scheduler      : typing.Any

  class SampleCommitteeEstimator(typing.NamedTuple):
    """Named tuple for sampling BERT."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : torchLMDataGenerator

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"

    self.logfile_path      = self.cache.path / "logs"

    self.is_validated      = False
    self.trained           = False
    l.logger().info("Active Committee config initialized in {}".format(self.cache.path))
    return

  def _ConfigTrainParams(self, 
                         data_generator: torchLMDataGenerator,
                         ) -> None:
    """
    Model parameter initialization for training and validation.
    """
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

    self.train = []

    for model in model_committee:
      m = model(self.config).to(self.pytorch.offset_device)
      if self.pytorch.num_nodes > 1:
        distrib.barrier()
        m = self.torch.nn.parallel.DistributedDataParallel(
          m,
          device_ids    = [self.pytorch.offset_device],
          output_device = self.pytorch.offset_device,
        )
      elif self.pytorch.num_gpus > 1:
        m = self.torch.nn.DataParallel(m)

      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = m,
        num_train_steps = self.num_train_steps,
        warmup_steps    = self.num_warmup_steps,
        learning_rate   = self.learning_rate,
      )
      self.train.append(
        ActiveCommittee.CommitteeEstimator(
          m, copy.deepcopy(data_generator), opt, lr_scheduler
        )
      )
    l.logger().info(self.GetShortSummary())
    return

  def Train(self, corpus, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    raise NotImplementedError

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
    return

  def Validate(self) -> None:
    raise NotImplementedError
    return

  def Sample(self) -> None:
    raise NotImplementedError
    return

  def saveCheckpoint(self, estimator):
    """
    Saves model, scheduler, optimizer checkpoints per epoch.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda x: self.ckpt_path / "{}-{}.pt".format(x, self.current_step)

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
        mf.write("train_step: {}\n".format(self.current_step))
    return

  def loadCheckpoint(self,
                     estimator: typing.Union[
                                  typing.TypeVar('torchBert.BertEstimator'),
                                  typing.TypeVar('torchBert.SampleBertEstimator')
                                ],
                     ) -> int:
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
          self.torch.load(ckpt_comp("model"))
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
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
          self.torch.load(ckpt_comp("model"), map_location=self.pytorch.device)
        )
      except RuntimeError:
        """
        Pytorch doesn't love loading a DataParallel checkpoint
        to a simple model. So, the following hack is needed
        to remove the 'module.' prefix from state keys.

        OR it might as well need the opposite. Transitioning from
        single to multiple GPUs will mean that 'module.' prefix is missing
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.torch.load(ckpt_comp("model")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        estimator.model.load_state_dict(new_state_dict)
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
