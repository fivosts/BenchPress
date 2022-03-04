"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import pathlib
import copy

from deeplearning.clgen.active_models import backends
from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.active_models.committee import config
from deeplearning.clgen.util.pytorch import torch
from deeplearning.clgen.util import logging as l



class ActiveCommittee(backends.BackendBase):

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
    data_generator : data_generator.Dataloader
    optimizer      : typing.Any
    scheduler      : typing.Any
    training_opts  : 'TrainingOpts'
    sha256         : str

  class SampleCommitteeEstimator(typing.NamedTuple):
    """Named tuple for sampling BERT."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : data_generator.Dataloader
    sha256         : str

  def __repr__(self):
    return "ActiveCommittee"

  def __init__(self, *args, **kwargs):

    super(ActiveCommittee, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.committee.random_seed)
    self.torch.cuda.manual_seed_all(self.config.committee.random_seed)

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"
    self.logfile_path      = self.cache.path / "logs"

    self.committee         = []

    self.is_validated      = False
    self.trained           = False
    l.logger().info("Active Committee config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, data_generator: data_generator.Dataloader) -> None:
    """
    Model parameter initialization.
    """

    self.committee_configs = config.CommitteeConfig.FromConfig(self.config.committee)

    self.validation_results_file = "val_results.txt"
    self.validation_results_path = os.path.join(str(self.logfile_path), self.validation_results_file)

    for cconfig in self.committee_configs:
      training_opts = ActiveCommittee.TrainingOpts(
        train_batch_size = cconfig.batch_size,
        learning_rate    = cconfig.learning_rate,
        num_warmup_steps = cconfig.num_warmup_steps,
        max_grad_norm    = cconfig.max_grad_norm,
        steps_per_epoch  = cconfig.steps_per_epoch,
        num_epochs       = cconfig.steps_per_epoch,
        num_train_steps  = cconfig.num_train_steps,
      )
      cm = models.Committee.FromConfig(cconfig)
      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = cm,
        num_train_steps = training_opts.num_train_steps,
        warmup_steps    = training_opts.num_warmup_steps,
        learning_rate   = training_opts.learning_rate,
      )
      self.committee.append(
        model          = cm,
        data_generator = data_generator,
        training_opts  = training_opts,
        sha256         = cconfig.sha256,
      )
      (self.ckpt_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
      (self.logfile_path / cconfig.sha256).mkdir(exist_ok = True, parents = True),
    l.logger().info(self.GetShortSummary())
    return

  def TrainMember(self, member: CommitteeEstimator) -> None:
    raise NotImplementedError
    return

  def Train(self, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    self._ConfigTrainParams()
    raise NotImplementedError

    ## config model array
    ## load checkpoint
    ## Data generator
    ## Schedulers, optimizers
    ## Train member

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
                                  typing.TypeVar('ActiveCommittee.CommitteeEstimator'),
                                  typing.TypeVar('ActiveCommittee.SampleCommitteeEstimator')
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
    if isinstance(estimator, ActiveCommittee.CommitteeEstimator):
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
