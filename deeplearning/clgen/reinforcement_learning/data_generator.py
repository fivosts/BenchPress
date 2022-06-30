"""
Memory replay buffer for reinforcement learning training.
"""
import typing
import numpy as np

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.features import extractor
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import pytorch

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

def StateToActionTensor(state: interactions.State, padToken: int, feat_padToken: int) -> typing.Dict[str, torch.Tensor]:
  """
  Pre-process state to tensor inputs for Action Deep QValues.
  """
  ids               = torch.LongTensor(state.encoded_code).unsqueeze(0)
  feat_ids          = torch.LongTensor(state.encoded_features).unsqueeze(0)
  ids_pad_mask      = ids      != padToken
  feat_ids_pad_mask = feat_ids != feat_padToken
  return {
    'input_ids'                        : ids,
    'target_features'                  : feat_ids,
    'input_ids_key_padding_mask'       : ids_pad_mask,
    'target_features_key_padding_mask' : feat_ids_pad_mask,
  }

def StateToTokenTensor(state         : interactions.State,
                       mask_idx      : int,
                       maskToken     : int,
                       padToken      : int,
                       feat_padToken : int,
                       ) -> typing.Dict[str, torch.Tensor]:
  """
  Pre-process state to 
  """
  seq_len      = len(state.encoded_code)
  feat_seq_len = len(state.encoded_features)

  masked_code  = np.concatenate((state.encoded_code[:mask_idx+1], [maskToken], state.encoded_code[mask_idx+1:]))
  masked_code  = torch.LongTensor(masked_code[:seq_len]).unsqueeze(0)
  enc_features = torch.LongTensor(state.encoded_features).unsqueeze(0)

  return {
    'encoder_input_ids'      : masked_code,
    'encoder_input_mask'     : masked_code != padToken,
    'encoder_position_ids'   : torch.arange(seq_len, dtype = torch.int64).unsqueeze(0),
    'decoder_feature_ids'    : enc_features,
    'decoder_feature_mask'   : enc_features != feat_padToken,
    'decoder_position_ids'   : torch.arange(feat_seq_len, dtype = torch.int64).unsqueeze(0),
    'encoder_input_features' : enc_features,
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
  