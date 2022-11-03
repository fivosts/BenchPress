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

    super(ExpectedErrorReduction, self).__init__(*args, **kwargs):
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

  def _ConfigModelParams(self,
                         data_generator : 'torch.utils.data.Dataset' = None,
                         is_sampling    : bool = False
                         ) -> None:
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
      cm = models.MLP.FromConfig(self.model_config)
      if not is_sampling:
        opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
          model           = cm,
          num_train_steps = self.training_opts.num_train_steps,
          warmup_steps    = self.training_opts.num_warmup_steps,
          learning_rate   = self.training_opts.learning_rate,
        )
      else:
        opt, lr_scheduler = None, None
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
