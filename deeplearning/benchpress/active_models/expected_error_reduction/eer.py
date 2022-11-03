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

from deeplearning.benchpress.active_models import backends

class ExpectedErrorReduction(backends.BackendBase):

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

    self.model = None

    self.is_validated = False
    self.is_trained   = False

    self.committee_samples = committee_database.CommitteeSamples(
      url        = "sqlite:///{}".format(str(self.sample_path / "samples.db")),
      must_exist = False,
    )
    self.sample_epoch = self.committee_samples.cur_sample_epoch
    l.logger().info("Active ExpectedErrorReduction config initialized in {}".format(self.cache_path))
    return

  def _ConfigModelParams(self,
                         data_generator : 'torch.utils.data.Dataset' = None,
                         is_sampling    : bool = False
                         ) -> None:
    """
    Model parameter initialization.
    """
    if not self.model:
      self.model_config = config.ModelConfig.FromConfig(
        self.config.expected_error_reduction, 
        self.downstream_task,
        self.config.num_train_steps
      )
      training_opts = QueryByCommittee.TrainingOpts(
        train_batch_size = self.model_config.batch_size,
        learning_rate    = self.model_config.learning_rate,
        num_warmup_steps = self.model_config.num_warmup_steps,
        max_grad_norm    = self.model_config.max_grad_norm,
        steps_per_epoch  = self.model_config.steps_per_epoch,
        num_epochs       = self.model_config.num_epochs,
        num_train_steps  = self.model_config.num_train_steps,
      )
      cm = models.MLP.FromConfig(idx, self.model_config)
      if not is_sampling:
        opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
          model           = cm,
          num_train_steps = training_opts.num_train_steps,
          warmup_steps    = training_opts.num_warmup_steps,
          learning_rate   = training_opts.learning_rate,
        )
      else:
        opt, lr_scheduler = None, None
      self.estimator = ExpectedErrorReduction.Estimator(
          model          = cm,
          data_generator = copy.deepcopy(data_generator),
          optimizer      = opt,
          scheduler      = lr_scheduler,
          training_opts  = training_opts,
          sha256         = self.model_config.sha256,
          config         = self.model_config,
          train_fn       = self.TrainNNMember if isinstance(cm, self.torch.nn.Module) else self.TrainUnsupervisedMember,
          sample_fn      = self.SampleNNMember if isinstance(cm, self.torch.nn.Module) else self.SampleUnsupervisedMember,
        )
      (self.ckpt_path / self.model_config.sha256).mkdir(exist_ok = True, parents = True),
      (self.logfile_path / self.model_config.sha256).mkdir(exist_ok = True, parents = True),
      l.logger().info(self.GetShortSummary())
    # for member in self.committee:
    #   self.committee_samples.add_member(
    #     member_id     = member.model.id,
    #     member_name   = member.config.name,
    #     type          = "supervised" if isinstance(member.model, self.torch.nn.Module) else "unsupervised",
    #     configuration = member.config.config,
    #   )
    return
