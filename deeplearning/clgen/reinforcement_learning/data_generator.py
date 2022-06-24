"""
Memory replay buffer for reinforcement learning training.
"""
import typing

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.features import extractor
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.util import pytorch

torch = pytorch.torch

def from_config(config            : reinforcement_learning_pb2.RLModel,
                feature_tokenizer : tokenizers.FeatureTokenizer,
                corpus            : corpuses.Corpus,
                ) -> "FeatureLoader":
  """
  Return the right torch dataloader based on configuration.
  """
  if config.HasField("train_set"):
    return CorpusFeatureLoader(config, corpus, feature_tokenizer)
  elif config.HasField("random"):
    return RandomFeatureLoader(config, feature_tokenizer)
  return

class CorpusFeatureLoader(torch.utils.Dataset):
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
      feats = extractor.RawToDictFeats(dp)
      for k, v in feats.items():
        fvec = self.feature_tokenizer.TokenizeFeatureVector(v, k, self.config.feature_sequence_length)
        self.dataset.append(
          {
            'input_features': torch.LongTensor(fvec),
            # 'input_features_mask': torch.LongTensor(fvec != self.feature_tokenizer.padToken),
            # 'input_features_key_padding_mask': None,
          }
        )
    return

class RandomFeatureLoader(torch.utils.data.Dataset):
  """
  Torch-based dataloading class for target feature vectors.
  """
  def __init__(self,
               config            : reinforcement_learning_pb2.RLModel,
               feature_tokenizer : tokenizers.FeatureTokenizer
               ):
    self.config = config
    self.feature_tokenizer = feature_tokenizer
    return
