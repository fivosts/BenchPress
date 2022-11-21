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
import os
import typing
import gdown
import shutil
import threading
import pathlib

from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.samplers import sample_observers
from deeplearning.benchpress.samplers import samplers
from deeplearning.benchpress.samplers import samples_database
from deeplearning.benchpress.models import language_models
from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.proto import sampler_pb2
from deeplearning.benchpress.proto import benchpress_pb2
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import logging as l

from absl import app, flags

FLAGS = flags.FLAGS

PRETRAINED_MODELS = {
  "base_opencl": {
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
  def FromID(cls, name: str = "base_opencl") -> "PreTrainedModel":
    if name not in PRETRAINED_MODELS:
      raise ValueError("Pre-trained model {} does not exist. Available models: {}".format(name, ', '.join([x for x in PRETRAINED_MODELS.keys()])))

    tdir = "/tmp/"
    if FLAGS.local_filesystem:
      tdir = FLAGS.local_filesystem

    config_path     = pathlib.Path(tdir) / "from_pretrained" / name/ "config.pbtxt"
    tokenizer_path  = pathlib.Path(tdir) / "from_pretrained" / name/ "tokenizer.pkl"
    checkpoint_path = pathlib.Path(tdir) / "from_pretrained" / name/ "model-0.pt"

    if environment.WORLD_RANK == 0:
      config_path.parent.mkdir(exist_ok = True, parents = True)

    if not config_path.exists():
      gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['config']), str(config_path))
    if not tokenizer_path.exists():
      gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['tokenizer']), str(tokenizer_path))
    if not checkpoint_path.exists():
      gdown.download("https://drive.google.com/uc?id={}".format(PRETRAINED_MODELS[name]['checkpoint']), str(checkpoint_path))

    model_config = pbutil.FromFile(config_path, benchpress_pb2.Instance()).language_model
    os.environ["PWD"] = str(config_path.parent)

    FLAGS.override_preprocessing = True
    FLAGS.override_encoding      = True
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
      if not (self.language_model.cache.path / "checkpoints" / "model-0.pt").exists():
        shutil.copyfile(checkpoint, self.language_model.cache.path / "checkpoints" / "model-0.pt")
      if not (self.language_model.cache.path / "checkpoints" / "checkpoint.meta").exists():
        with open(self.language_model.cache.path / "checkpoints" / "checkpoint.meta", 'w') as outf:
          outf.write("train_step: 0")
    if environment.WORLD_SIZE > 1:
      distrib.barrier()
    return

  def Sample(self,
             prompt: str,
             batch_size: int = 1,
             temperature: float = 0.6,
             sample_workload_size: int = 1,
             sample_indices_limit: int = None,
             print_samples: bool = True,
             seed: int = None,
             ) -> typing.Tuple[str, samples_database.Sample]:
    """
    Get a string input, tokenize and sample the backend online for a full code.

    Args:
      prompt:
        String input to the language model.
      batch_size:
        Batch size for model inference.
      temperature:
        Sampling temperature
      sample_workload_size:
        How many batches to generate.
      sample_indices_limit:
        Add a limit to how many tokens BenchPress will generate for a hole.
        By default BenchPress generates tokens until it thinks a sequence is complete
        ([ENDHOLE] is generated). By setting this value, generation loop will be killed
        after surpassing this threshold.
    """
    FLAGS.sample_workload_size = sample_workload_size
    if sample_indices_limit is not None:
      FLAGS.sample_indices_limit = sample_indices_limit

    self.language_model.Create()
    if pretrained.language_model.backend.pytorch.num_gpus == 0:
      l.logger().warn("No GPUs detected. This process is going to be *very* slow on the CPU.")
    if "[START]" in prompt or "[END]" in prompt:
      l.logger().error("Do not add [START] and [END] manually. They will be added automatically by the tokenizer.")
      return ""
    prompt = "[START]" + prompt + "[END]"
    test_sampler = self.getTestSampler(prompt, batch_size, temperature, self.language_model.config.architecture.max_position_embeddings)
    obs = [sample_observers.InMemorySampleSaver()]
    if print_samples:
      obs.append(sample_observers.PrintSampleObserver())
    self.language_model.Sample(test_sampler, obs, num_batches = 1, seed = seed)
    return [opencl.ClangFormat(x.text) for x in obs[0].samples], obs[0].samples

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
        "temperature_micros: {}".format(int(temperature * 10e6)),
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
