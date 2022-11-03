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

    self.committee    = None

    self.is_validated = False
    self.is_trained   = False

    self.committee_samples = committee_database.CommitteeSamples(
      url        = "sqlite:///{}".format(str(self.sample_path / "samples.db")),
      must_exist = False,
    )
    self.sample_epoch = self.committee_samples.cur_sample_epoch
    l.logger().info("Active ExpectedErrorReduction config initialized in {}".format(self.cache_path))
    return
