"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib
import pickle
import math
import functools
import typing
import tqdm
import multiprocessing
import time
import numpy as np

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import grewe
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import http_server
from deeplearning.clgen.util import socket_server
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.models.torch_bert.data_generator import JSON_to_ActiveSample
from deeplearning.clgen.models.torch_bert.data_generator import ActiveSample_to_JSON

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "server_tokenizer",
  None,
  "Set path for tokenizer to be used by downstream server."
)

flags.DEFINE_string(
  "server_cldrive_cache",
  None,
  "Set path for cldrive_cache to be used by downstream server."
)

flags.DEFINE_boolean(
  "only_optimal_gsize",
  False,
  "If True, only the best matching global size to transferred_bytes will be executed. Otherwise, everything."
)

def ExtractorWorker(cldrive_entry: cldrive.CLDriveSample, fspace: str):
  """
  Worker that extracts features and buffers cldrive entry, to maintain consistency
  among multiprocessed data.
  """
  features = extractor.ExtractFeatures(cldrive_entry.source, [fspace])
  if fspace in features and features[fspace]:
    return features[fspace], cldrive_entry
  return None

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str, corpus_path: pathlib.Path, random_seed: int, **kwargs) -> "DownstreamTask":
    return TASKS[task](corpus_path, random_seed, kwargs)

  def __init__(self, name: str, random_seed: int) -> None:
    self.name        = name
    self.random_seed = random_seed
    return

  def step_generation(self, candidates: typing.List['ActiveSample']) -> None:
    return

  def saveCheckpoint(self) -> None:
    raise NotImplementedError("Abstract Class")

  def loadCheckpoint(self) -> None:
    raise NotImplementedError("Abstract Class")

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  This class is responsible to fetch the raw data and act as a tokenizer
  for the data. Reason is, the data generator should be agnostic of the labels.
  """
  @property
  def input_size(self) -> int:
    return 4
  
  @property
  def static_features_size(self) -> int:
    return len(grewe.KEYS)

  @property
  def runtime_features_size(self) -> int:
    return 2

  @property
  def input_labels(self) -> typing.List[str]:
    return [
      "tr_bytes/(comp+mem)",
      "coalesced/mem",
      "localmem/(mem+wgsize)",
      "comp/mem"
    ]

  @property
  def output_size(self) -> int:
    return 2

  @property
  def output_labels(self) -> typing.Tuple[str, str]:
    return ["CPU", "GPU"]

  @property
  def output_ids(self) -> typing.Tuple[str, str]:
    return [0, 1]

  @property
  def feature_space(self) -> str:
    return "GreweFeatures"

  def __init__(self,
               corpus_path   : pathlib.Path,
               random_seed   : int,
               use_as_server : bool = False,
               ) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive", random_seed)
    self.corpus_path = corpus_path
    self.corpus_db   = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus_path)), must_exist = True)
    if use_as_server:
      self.setup_server()
    else:
      ## Setup random seed np random stuff
      self.rand_generator = np.random
      self.rand_generator.seed(random_seed)
      self.gen_bounds = {
        'comp'             : (1, 300),
        'rational'         : (0, 50),
        'mem'              : (1, 50),
        'localmem'         : (0, 50),
        'coalesced'        : (0, 10),
        'atomic'           : (0, 10),
        'transferred_bytes': (1, 31), # 2**pow,
        'local_size'       : (1, 8),  # 2**pow,
      }
    return

  def __repr__(self) -> str:
    return "GrewePredictive"

  def setup_server(self) -> None:
    """
    In server mode, initialize the serving process.
    """
    if environment.WORLD_RANK == 0:
      self.cl_proc, self.work_flag, self.read_queue, self.write_queues, self.reject_queues = http_server.start_server_process()
    return

  def setup_dataset(self, num_train_steps: int = None) -> None:
    """
    Fetch data and preprocess into corpus for Grewe's predictive model.
    """
    self.dataset = []
    data = [x for x in self.corpus_db.get_valid_data(dataset = "GitHub")]
    pool = multiprocessing.Pool()
    it = pool.imap_unordered(functools.partial(ExtractorWorker, fspace = "GreweFeatures"), data)
    idx = 0
    try:
      loop = tqdm.tqdm(it, total = len(data), desc = "Grewe corpus setup", leave = False) if environment.WORLD_RANK == 0 else it
      for dp in loop:
        if dp:
          feats, entry = dp
          self.dataset.append(
            (
              self.InputtoEncodedVector(feats, entry.transferred_bytes, entry.local_size),
              [self.TargetLabeltoID(entry.status)]
            )
          )
          idx += 1
        # if idx >= 100:
          # break
      pool.close()
    except Exception as e:
      pool.terminate()
      raise e
    # pool.terminate()
    if num_train_steps:
      self.data_generator = data_generator.ListTrainDataloader(self.dataset[:num_train_steps])
    else:
      self.data_generator = data_generator.ListTrainDataloader(self.dataset)
    return

  def CollectSingleRuntimeFeature(self,
                                  sample: 'ActiveSample',
                                  tokenizer: 'tokenizers.TokenizerBase',
                                  store_rejects: bool = False,
                                  ) -> typing.Tuple[typing.List['ActiveSample'], typing.List['ActiveSample']]:
    """
    Overloaded function to compute runtime features for a single instance.
    """
    def create_sample(s: 'ActiveSample', code: str, trb: int, gs: int) -> typing.List['ActiveSample']:
      nrfeats = s.runtime_features
      nrfeats['transferred_bytes'] = trb
      nrfeats['global_size'] = int(2**gs)
      cached = self.corpus_db.update_and_get(
        code,
        "BenchPress_AL_Grewe",
        global_size = nrfeats['global_size'],
        local_size  = nrfeats['local_size'],
        num_runs    = 1000,
        timeout     = 60,
      )
      nrfeats['label'] = cached.status
      return s._replace(runtime_features = nrfeats)

    exp_tr_bytes = sample.runtime_features['transferred_bytes']
    local_size   = sample.runtime_features['local_size']
    found        = False
    gsize        = max(1, math.log2(local_size))
    prev         = math.inf
    code         = tokenizer.ArrayToCode(sample.sample)
    new_samples  = []
    rejects      = []
    while not found and gsize <= 20:
      sha256 = crypto.sha256_str(code + "BenchPress_AL_Grewe" + str(2**gsize) + str(local_size))
      if sha256 in self.corpus_db.status_cache:
        cached = self.corpus_db.get_entry(code, "BenchPress_AL_Grewe", int(2**gsize), int(local_size))
      else:
        cached = self.corpus_db.update_and_get(
          code,
          "BenchPress_AL_Grewe",
          global_size = int(2**gsize),
          local_size  = int(local_size),
          num_runs    = 1000,
          timeout     = 60,
        )
      if cached.status in {"CPU", "GPU"}:
        tr_bytes = cached.transferred_bytes
        if not FLAGS.only_optimal_gsize:
          new_samples.append(
            create_sample(
              s    = sample,
              code = code,
              trb  = tr_bytes,
              gs   = gsize
            )
          )
      else:
        if store_rejects:
          rejects.append(
            create_sample(
              s    = sample,
              code = code,
              trb  = exp_tr_bytes,
              gs   = gsize,
            )
          )
        tr_bytes = None
      if tr_bytes:
        if tr_bytes < exp_tr_bytes:
          gsize += 1
          prev = tr_bytes
        else:
          found = True
          if abs(exp_tr_bytes - tr_bytes) > abs(exp_tr_bytes - prev):
            gsize -= 1
            tr_bytes  = abs(exp_tr_bytes - prev)
          else:
            tr_bytes  = abs(exp_tr_bytes - tr_bytes)
      else:
        gsize += 1
    if FLAGS.only_optimal_gsize:
      new_samples = [create_sample(sample, code, tr_bytes if found else exp_tr_bytes, gsize)]
    return new_samples, rejects

  def ServeRuntimeFeatures(self, tokenizer: 'tokenizers.TokenizerBase') -> None:
    """
    In server mode, listen to the read queue, collect runtime features,
    append to local cache and publish to write queue for the client to fetch.
    This has been easily implemented only for HTTP server and not socket.
    """
    try:
      while self.cl_proc.is_alive():
        if not self.read_queue.empty():
          self.work_flag.value = True
          source, serialized   = self.read_queue.get()
          sample   = JSON_to_ActiveSample(serialized)
          ret, rej = self.CollectSingleRuntimeFeature(sample, tokenizer, store_rejects = True)
          for x in ret:
            self.write_queues[source].append(ActiveSample_to_JSON(x))
          for x in rej:
            self.reject_queues[source].append(ActiveSample_to_JSON(x))
          self.work_flag.value = False
        else:
          time.sleep(1)
    except KeyboardInterrupt:
      pass
    return

  def CollectRuntimeFeatures(self,
                             samples   : typing.List['ActiveSample'],
                             top_k     : int,
                             tokenizer : 'tokenizers.TokenizerBase',
                            ) -> typing.List['ActiveSample']:
    """
    Collect the top_k samples that can run on CLDrive and set their global size
    to the appropriate value so it can match the transferred bytes.
    """
    if FLAGS.use_http_server:
      new_samples = []
      while int(http_server.client_status_request()[1]) != 200:
        batch = http_server.client_get_request()
        for ser in batch:
          obj = JSON_to_ActiveSample(ser)
          new_samples.append(obj)
        time.sleep(5)
      if top_k != -1:
        return sorted([x for x in new_samples if x.runtime_features['label']], key = lambda x: x.score)[:top_k]
      else:
        return sorted([x for x in new_samples if x.runtime_features['label']], key = lambda x: x.score)
    else:
      new_samples = []
      total = 0
      for sample in sorted(samples, key = lambda x: x.score):
        ret = self.CollectSingleRuntimeFeature(sample, tokenizer)
        for s in ret:
          if s.runtime_features['label'] in {"CPU", "GPU"}:
            total += 1
            new_samples.append(s)
          if top_k != -1 and total >= top_k:
            return new_samples
      return new_samples

  def UpdateDataGenerator(self,
                          new_samples: typing.List['ActiveSample'],
                          top_k: int,
                          tokenizer: 'tokenizers.TokenizerBase',
                          ) -> data_generator.ListTrainDataloader:
    """
    Collect new generated samples, find their runtime features and processs to a torch dataset.
    """
    new_samples = self.CollectRuntimeFeatures(new_samples, top_k, tokenizer)
    updated_dataset = [
      (
        self.InputtoEncodedVector(entry.features,
                                  entry.runtime_features['transferred_bytes'],
                                  entry.runtime_features['local_size']
                                  ),
        [self.TargetLabeltoID(entry.runtime_features['label'])]
      ) for entry in new_samples
    ]
    if len(updated_dataset) == 0:
      l.logger().warn("Update dataset is empty.")
    return updated_dataset, data_generator.ListTrainDataloader(updated_dataset, lazy = True)

  def UpdateTrainDataset(self, updated_dataloader: data_generator.ListTrainDataloader) -> None:
    """
    After active learner has been updated, store updated samples to original train dataset.
    """
    self.data_generator = self.data_generator + updated_dataloader
    return

  def sample_space(self, num_samples: int = 512) -> data_generator.DictPredictionDataloader:
    """
    Go fetch Grewe Predictive model's feature space and randomly return num_samples samples
    to evaluate. The predictive model samples are mapped as a value to the static features
    as a key.
    """
    samples = []
    samples_hash = set()
    for x in range(num_samples):
      fvec = {
        k: self.rand_generator.randint(self.gen_bounds[k][0], self.gen_bounds[k][1])
        for k in grewe.KEYS if k not in {"F2:coalesced/mem", "F4:comp/mem"}
      }
      try:
        fvec['F2:coalesced/mem'] = fvec['coalesced'] / fvec['mem']
      except ZeroDivisionError:
        fvec['F2:coalesced/mem'] = 0.0
      try:
        fvec['F4:comp/mem'] = fvec['comp'] / fvec['mem']      
      except ZeroDivisionError:
        fvec['F4:comp/mem'] = 0.0
      transferred_bytes = 2**self.rand_generator.randint(self.gen_bounds['transferred_bytes'][0], self.gen_bounds['transferred_bytes'][1])
      local_size        = 2**self.rand_generator.randint(self.gen_bounds['local_size'][0], self.gen_bounds['local_size'][1])
      inp_ids           = self.InputtoEncodedVector(fvec, transferred_bytes, local_size)
      if str(inp_ids) not in samples_hash:
        samples.append(
          {
            'static_features'  : self.StaticFeatDictToVec(fvec),
            'runtime_features' : [transferred_bytes, local_size],
            'input_ids'        : inp_ids,
          }
        )
        samples_hash.add(str(inp_ids))
    return data_generator.DictPredictionDataloader(samples)

  def step_generation(self, candidates: typing.List['ActiveSample']) -> None:
    """
    End of LM generation's epoch hook.
    """
    if FLAGS.use_http_server:
      serialized = []
      for cand in candidates:
        serialized.append(
          ActiveSample_to_JSON(cand)
        )
      http_server.client_put_request(serialized)
    return

  def StaticFeatDictToVec(self, static_feats: typing.Dict[str, float]) -> typing.List[float]:
    """
    Process grewe static features dictionary into list of floats to be passed as tensor.
    """
    return [static_feats[key] for key in grewe.KEYS]

  def VecToStaticFeatDict(self, feature_values: typing.List[float]) -> typing.Dict[str, float]:
    """
    Process float vector of feature values to dictionary of features.
    """
    return {key: val for key, val in zip(grewe.KEYS, feature_values)}

  def VecToRuntimeFeatDict(self, runtime_values: typing.List[int]) -> typing.Dict[str, int]:
    """
    Process runtime int values to runtime features dictionary.
    """
    trb, ls = runtime_values
    return {
      'transferred_bytes' : trb,
      'local_size'        : ls,
    }

  def VecToInputFeatDict(self, input_values: typing.List[float]) -> typing.Dict[str, float]:
    """
    Convert to dictionary of predictive model input features.
    """
    return {
      "tr_bytes/(comp+mem)"   : input_values[0],
      "coalesced/mem"         : input_values[1],
      "localmem/(mem+wgsize)" : input_values[2],
      "comp/mem"              : input_values[3],
    }

  def InputtoEncodedVector(self,
                           static_feats      : typing.Dict[str, float],
                           transferred_bytes : int,
                           local_size       : int,
                           ) -> typing.List[float]:
    """
    Encode consistently raw features to Grewe's predictive model inputs.
    """
    try:
      i1 = transferred_bytes / (static_feats['comp'] + static_feats['mem'])
    except ZeroDivisionError:
      i1 = 0.0
    try:
      i2 = static_feats['coalesced'] / static_feats['mem']
    except ZeroDivisionError:
      i2 = 0.0
    try:
      i3 = static_feats['localmem'] / (static_feats['mem'] * local_size)
    except ZeroDivisionError:
      i3 = 0.0
    try:
      i4 = static_feats['comp'] / static_feats['mem']
    except ZeroDivisionError:
      i4 = 0.0
    return [i1, i2, i3, i4]

  def TargetIDtoLabels(self, id: int) -> str:
    """
    Integer ID to label of predictive model.
    """
    return {
      0: "CPU",
      1: "GPU",
    }[id]

  def TargetLabeltoID(self, label: str) -> int:
    """
    Predictive label to ID.
    """
    return {
      "CPU": 0,
      "GPU": 1,
    }[label]

  def TargetLabeltoEncodedVector(self, label: str) -> typing.List[int]:
    """
    Label to target vector.
    """
    return {
      "CPU": [1, 0],
      "GPU": [0, 1],
    }[label]

  def saveCheckpoint(self) -> None:
    """
    Store data generator.
    """
    return

TASKS = {
  "GrewePredictive": GrewePredictive,
}

def main(*args, **kwargs) -> None:
  if FLAGS.server_tokenizer is None:
    raise ValueError("Please define --server_tokenizer")
  if FLAGS.server_cldrive_cache is None:
    raise ValueError("Please define --server_cldrive_cache")
  tokenizer_path = pathlib.Path(FLAGS.server_tokenizer).resolve()
  cldrive_cache  = pathlib.Path(FLAGS.server_cldrive_cache).resolve()
  if not tokenizer_path.exists():
    raise FileNotFoundError(tokenizer_path)
  if not cldrive_cache.exists():
    raise FileNotFoundError(cldrive_cache)
  if not FLAGS.use_http_server and not FLAGS.use_socket_server:
    raise ValueError("This booting point is supposed to work as server. Set your flags appropriately.")
  tokenizer = tokenizers.TokenizerBase.FromFile(tokenizer_path)
  task = DownstreamTask.FromTask("GrewePredictive", cldrive_cache, 0, use_as_server = True)
  task.ServeRuntimeFeatures(tokenizer)
  return

if __name__ == "__main__":
  app.run(main)
  exit()
