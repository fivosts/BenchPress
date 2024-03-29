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
Memory replay buffer for reinforcement learning training.
"""
import typing
import numpy as np

from deeplearning.benchpress.reinforcement_learning import interactions
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.corpuses import corpuses
from deeplearning.benchpress.features import extractor
from deeplearning.benchpress.proto import reinforcement_learning_pb2
from deeplearning.benchpress.util import pytorch

torch = pytorch.torch

def from_config(config            : reinforcement_learning_pb2.RLModel,
                feature_tokenizer : tokenizers.FeatureTokenizer,
                corpus            : corpuses.Corpus,
                ) -> torch.utils.data.Dataset:
  """
  Return the right torch dataloader based on configuration.
  """
  if config.HasField("train_set"):
    return CorpusFeatureLoader(config, corpus, feature_tokenizer)
  elif config.HasField("random"):
    return RandomFeatureLoader(config, feature_tokenizer)
  return

def StateToActionTensor(state         : interactions.State,
                        padToken      : int,
                        feat_padToken : int,
                        batch_size    : int,
                        ) -> typing.Dict[str, torch.Tensor]:
  """
  Pre-process state to tensor inputs for Action Deep QValues.
  """
  seq_len      = len(state.encoded_code)
  feat_seq_len = len(state.encoded_features)

  src_ids  = torch.LongTensor(state.encoded_code).unsqueeze(0).repeat(batch_size, 1)
  src_mask = src_ids != padToken
  src_pos  = torch.arange(seq_len, dtype = torch.int64).unsqueeze(0).repeat(batch_size, 1)

  feat_ids  = torch.LongTensor(state.encoded_features).unsqueeze(0).repeat(batch_size, 1)
  feat_mask = feat_ids != feat_padToken
  feat_pos  = torch.arange(feat_seq_len, dtype = torch.int64).unsqueeze(0).repeat(batch_size, 1)
  return {
    'encoder_feature_ids'  : feat_ids,
    'encoder_feature_mask' : feat_mask,
    'encoder_position_ids' : feat_pos,
    'decoder_input_ids'    : src_ids,
    'decoder_input_mask'   : src_mask,
    'decoder_position_ids' : src_pos,
  }

def StateToTokenTensor(state          : interactions.State,
                       mask_idx       : int,
                       maskToken      : int,
                       padToken       : int,
                       feat_padToken  : int,
                       batch_size     : int,
                       replace_token  : bool = False,
                       ) -> typing.Dict[str, torch.Tensor]:
  """
  Pre-process state to 
  """
  seq_len      = len(state.encoded_code)
  feat_seq_len = len(state.encoded_features)

  if replace_token:
    masked_code = state.encoded_code
    masked_code[mask_idx] = maskToken
  else:
    masked_code = np.concatenate((state.encoded_code[:mask_idx+1], [maskToken], state.encoded_code[mask_idx+1:]))
  masked_code  = torch.LongTensor(masked_code[:seq_len]).unsqueeze(0).repeat(batch_size, 1)
  enc_features = torch.LongTensor(state.encoded_features).unsqueeze(0).repeat(batch_size, 1)

  return {
    'encoder_feature_ids'    : enc_features,
    'encoder_feature_mask'   : enc_features != feat_padToken,
    'encoder_position_ids'   : torch.arange(feat_seq_len, dtype = torch.int64).unsqueeze(0).repeat(batch_size, 1),
    'decoder_input_ids'      : masked_code,
    'decoder_input_mask'     : masked_code != padToken,
    'decoder_position_ids'   : torch.arange(seq_len, dtype = torch.int64).unsqueeze(0).repeat(batch_size, 1),
  }

class CorpusFeatureLoader(torch.utils.data.Dataset):
  """
  Dataloading from language model's training corpus.
  """
  def __init__(self,
               config: reinforcement_learning_pb2.RLModel,
               corpus: corpuses.Corpus,
               feature_tokenizer: tokenizers.FeatureTokenizer
               ):
    self.config = config
    self.data = corpus.GetTrainingFeatures()
    self.feature_tokenizer = feature_tokenizer
    self.setup_dataset()
    return

  def __len__(self) -> int:
    return len(self.dataset)
  
  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:
    return
  
  def setup_dataset(self) -> typing.List[typing.Dict[str, torch.Tensor]]:
    """Process raw feature vectors to processed dataset."""
    self.dataset = []
    for dp in self.data:
      for k, v in dp.items():
        if v:
          fvec = self.feature_tokenizer.TokenizeFeatureVector(v, k, self.config.agent.action_qv.feature_sequence_length)
          self.dataset.append(
            {
              'input_features': torch.LongTensor(fvec),
              'input_features_key_padding_mask': torch.LongTensor(fvec != self.feature_tokenizer.padToken),
            }
          )
    return

class RandomFeatureLoader(torch.utils.data.Dataset):
  """
  Torch-based dataloading class for target feature vectors.
  """
  def __init__(self,
               config            : reinforcement_learning_pb2.RLModel,
               feature_tokenizer : tokenizers.FeatureTokenizer,
               ):
    self.config = config
    self.feature_tokenizer = feature_tokenizer
    return

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int) -> typing.Dict[str, torch.Tensor]:
    return
  