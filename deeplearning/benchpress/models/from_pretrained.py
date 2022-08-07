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
import gdown
import shutil
import threading
import pathlib
import pickle
import sys

from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.samplers import sample_observers
from deeplearning.benchpress.samplers import samplers
from deeplearning.benchpress.models import builders
from deeplearning.benchpress.models import language_models
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.proto import sampler_pb2
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import logging as l

from absl import app, flags

FLAGS = flags.FLAGS

PRETRAINED_MODELS = {
  "base_benchpress": {
    'config'     : "1Cr9I4b5mSZJgX9LqtC_38WRfEDkyJ9WO",
    'tokenizer'  : "14ZPYFgL-XT_Fknwmgp6nOatvLFS67QM1",
    'checkpoint' : "1ncwxzR23_a6IQqt4F4gIgTeduggD_N9w",
  }
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

    config_path     = pathlib.Path(FLAGS.local_filesystem) / "from_pretrained" / "config.pbtxt"
    tokenizer_path  = pathlib.Path(FLAGS.local_filesystem) / "from_pretrained" / "tokenizer.pkl"
    checkpoint_path = pathlib.Path(FLAGS.local_filesystem) / "from_pretrained" / "model-0.pt"

    gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['config']), str(config_path))
    gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['tokenizer']), str(tokenizer_path))
    gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['checkpoint']), str(checkpoint_path))

    model_config = model_pb2.Model()
    model_config.CopyFrom(builders.AssertIsBuildable(model_config))

    FLAGS.override_preprocessing = True
    FLAGS.override_encoding = True
    return PreTrainedModel(model_config, tokenizer_path, checkpoint_path)

  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    return self.language_model.tokenizer

  def __init__(self,
               config          : model_pb2.Model,
               tokenizer_path  : tokenizers.TokenizerBase,
               checkpoint      : pathlib.Path,
               ):
    """
    Instantiate a model.

    Args:
      config: A Model message.

    Raises:
      TypeError: If the config argument is not a Model proto.
      UserError: In case on an invalid config.
    """
    self.language_model = language_models.Model(config)
    if environment.WORLD_RANK == 0:
      if not self.language_model.corpus.tokenizer_path.exists():
        shutil.copyfile(tokenizer_path, self.language_model.corpus.tokenizer_path)
      if not (self.language_model.cache.path / "checkpoints" / "backup_tokenizer.pkl").exists():
        shutil.copyfile(tokenizer_path, self.language_model.cache.path / "checkpoints" / "backup_tokenizer.pkl")
      if not (self.language_model.cache.path / "checkpoints" / "model-0").exists():
        shutil.copyfile(checkpoint, self.language_model.cache.path / "checkpoints" / "model-0")
      if not (self.language_model.cache.path / "checkpoints" / "checkpoint.meta").exists():
        with open(self.language_model.cache.path / "checkpoints" / "checkpoint.meta") as outf:
          outf.write("train_step: 0")
    distrib.barrier()
    return

  def Sample(self, prompt: str, batch_size: int = 1, temperature: float = 0.7) -> str:
    """
    Get a string input, tokenize and sample the backend online for a full code.
    """
    self.language_model.Create()
    encoded = self.tokenizer.TokenizeString(prompt)
    if self.tokenizer.holeToken not in encoded:
      l.logger().error("[HOLE] token not found in prompt. BenchPress needs this meta token to perform infilling.")
      return ""
    if len(np.where(encoded == self.tokenizer.holeToken)) > 1:
      l.logger().warn("BenchPress has been trained for single [HOLE] prompts only. Not sure how accurate it will be for multiple holes at the same time.")
    if self.tokenizer.startToken or self.tokenizer.endToken in encoded:
      l.logger().error("Do not add [START] and [END] manually. They will be added automatically by the tokenizer.")
      return ""
    encoded = [self.tokenizer.startToken] + encoded + [self.tokenizer.endToken]
    if len(encoded) > self.language_model.config.architecture.max_position_embeddings:
      l.logger().error("Length of prompt {} surpasses max position embeddings {}!".format(len(encoded), self.language_model.config.architecture.max_position_embeddings))
      return
    encoded = encoded + [self.tokenizer.padToken] * (self.language_model.config.architecture.max_position_embeddings - len(encoded))
    test_sampler = self.getTestSampler(prompt, batch_size, temperature, self.language_model.config.architecture.max_position_embeddings)
    obs = sample_observers.InMemorySampleSaver()
    self.language_model.Sample(test_sampler, obs)
    return obs.samples

  def getTestSampler(self,
                     prompt          : str,
                     batch_size      : int,
                     temperature     : float,
                     sequence_length : int
                     ) -> samplers.Sampler:
    sampler_str = [
        "start_text: \"{}\"".format(prompt),
        "batch_size: {}".format(batch_size),
        "sequence_length: {}".format(sequence_length),
        "temperature_micros: {}".format(temperature * 10e6),
    ]
    mock_config = pbutil.FromString('\n'.join(sampler_str), sampler_pb2.Sampler())
    sampler = samplers.Sampler(mock_config, sample_db_name = None)
    if sampler.isFixedStr:
      sampler.Specialize(self.tokenizer)
    return sampler

def main(*args, **kwargs) -> None:
  return

def boot() -> None:
  app.run(main)
  return

th = threading.Thread(target = boot)
th.setDaemon = True
th.start()
