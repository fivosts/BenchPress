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
import math
from deeplearning.benchpress.util.pytorch import torch

def swish(x):
  return x * torch.sigmoid(x)


def _gelu_python(x):
  """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
  """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
  """
  return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if torch.__version__ < "1.4.0":
  gelu = _gelu_python
else:
  gelu = torch.nn.functional.gelu


def gelu_fast(x):
  return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {
  "relu": torch.nn.functional.relu,
  "swish": swish,
  "gelu": gelu,
  "tanh": torch.tanh,
  "gelu_new": gelu_new,
  "gelu_fast": gelu_fast,
}


def get_activation(activation_string):
  if activation_string in ACT2FN:
    return ACT2FN[activation_string]
  else:
    raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))
