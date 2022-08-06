# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
Instance of pre-trained BenchPress Instances.

In this mode, a checkpoint is fetched online and the model is only used
for interactive sampling.
"""
import typing
import threading
import sys

from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.util import logging as l

from absl import app, flags

FLAGS = flags.FLAGS

PRETRAINED_MODELS = {
  "base_benchpress", ""
}

class PreTrainedModel(object):
  """
  Pre-trained instance wrapper for online checkpoint fetching
  and sampling.
  """
  @classmethod
  def from_pretrained(name: str = "base_benchpress") -> "PreTrainedModel":
    if name not in PRETRAINED_MODELS:
      raise ValueError("Pre-trained model {} does not exist. Available models: {}".format(name, ', '.join([x for x in PRETRAINED_MODELS.keys()])))
    url = PRETRAINED_MODELS[name]
    return
  
  def __init__(self, config: model_pb2.Model):
    return

def main(*args, **kwargs) -> None:
  return

def boot() -> None:
  app.run(main)
  return

th = threading.Thread(target = boot)
th.setDaemon = True
th.start()
