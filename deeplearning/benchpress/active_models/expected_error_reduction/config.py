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
Config setup for expected error reduction active learner.
"""
import typing
import pathlib

from deeplearning.benchpress.active_models import downstream_tasks
from deeplearning.benchpress.proto import active_learning_pb2
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import crypto

def AssertConfigIsValid(config: active_learning_pb2.ActiveLearner.ExpectedErrorReduction) -> None:
  """
  Parse proto description and check for validity.
  """
  if config.HasField("mlp"):
    tl = 0
    pbutil.AssertFieldIsSet(config.mlp, "initial_learning_rate_micros")
    pbutil.AssertFieldIsSet(config.mlp, "batch_size")
    pbutil.AssertFieldIsSet(config.mlp, "num_warmup_steps")
    for l in config.mlp.layer:
      if l.HasField("embedding"):
        pbutil.AssertFieldIsSet(l.embedding, "num_embeddings")
        pbutil.AssertFieldIsSet(l.embedding, "embedding_dim")
      elif l.HasField("linear"):
        pbutil.AssertFieldIsSet(l.linear, "in_features")
        pbutil.AssertFieldIsSet(l.linear, "out_features")
      elif l.HasField("dropout"):
        pbutil.AssertFieldIsSet(l.dropout, "p")
      elif l.HasField("layer_norm"):
        pbutil.AssertFieldIsSet(l.layer_norm, "normalized_shape")
        pbutil.AssertFieldIsSet(l.layer_norm, "eps")
      elif l.HasField("act_fn"):
        pbutil.AssertFieldIsSet(l.act_fn, "fn")
      else:
        raise AttributeError(l)
      tl += 1
    assert tl > 0, "Model is empty. No layers found."
  return