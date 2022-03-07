"""
Here all the committee members are defined.
"""
import math
import typing
import numpy as np

from deeplearning.clgen.active_models.committee import config
from deeplearning.clgen.models.torch_bert import activations
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util.pytorch import torch

from deeplearning.clgen.util import logging as l

def mish(x):
  return x * torch.tanh(torch.nn.functional.softplus(x))

ACT2FN = {
  "gelu"     : activations.gelu,
  "relu"     : torch.nn.functional.relu,
  "swish"    : activations.swish,
  "gelu_new" : activations.gelu_new,
  "mish"     : mish,
  "softmax"  : torch.nn.Softmax()
}

class CommitteeModels(torch.nn.Module):
  """
  Abstract representation of model committee.
  """
  @classmethod
  def FromConfig(cls, id: int, config: config.ModelConfig) -> "CommitteeModels":
    print(config)
    return {
      'MLP': MLP,
    }[config.name](id, config.layer_config)

  def __init__(self, id: int):
    super(CommitteeModels, self).__init__()
    self.id = id
    return

  def forward(self,
              input_ids   : torch.Tensor,
              target_ids  : torch.Tensor,
              is_sampling : bool = False
              ) -> typing.Dict[str, torch.Tensor]:
    raise NotImplementedError("Abstract class.")

class MLP(CommitteeModels):
  """
  A modular MLP model that supports Linear, Dropout, LayerNorm and activations.
  """
  def __init__(self, id: int, config: typing.List):
    super(MLP, self).__init__(id)
    self.config = config
    self.layers = []

    layers = {
      'Linear'    : torch.nn.Linear,
      'Dropout'   : torch.nn.Dropout,
      'LayerNorm' : torch.nn.LayerNorm,
    }
    layers.update(ACT2FN)
    self.layers = torch.nn.ModuleList([layers[layer[0]](**layer[1]) for layer in config])
    return

  def calculate_loss(self,
                     outputs: torch.Tensor,
                     target_ids: torch.Tensor,
                     ) -> torch.Tensor:
    """
    Categorical cross-entropy function.
    """
    print(outputs.shape)
    print(target_ids.shape)
    ## Calculate categorical label loss.
    loss_fn = torch.nn.CrossEntropyLoss()
    label_loss = loss_fn(outputs.to(torch.float32), target_ids.squeeze(1))

    ## Calculate top-1 accuracy of predictions across batch.
    hits, total = 0, int(outputs.size(0))
    probs       = self.softmax(outputs)
    outputs     = torch.argmax(probs, dim = 1)
    for out, target in zip(outputs, target_ids):
      if out == target:
        hits += 1
    return label_loss, torch.FloatTensor(hits / total)

  def forward(self,
              input_ids   : torch.Tensor,
              target_ids  : torch.Tensor = None,
              is_sampling : bool = False
              ) -> torch.Tensor:

    device = input_ids.get_device()
    device = device if device >= 0 else 'cpu'

    out = input_ids
    for layer in self.layers:
      out = layer(out)

    if not is_sampling:
      total_loss, batch_accuracy = self.calculate_loss(out, target_ids)
      return {
        'total_loss'   : total_loss,
        'accuracy'     : batch_accuracy.to(device),
        'output_label' : torch.argmax(out)
      }
    else:
      return {
        'output_label' : torch.argmax(out)
      }
