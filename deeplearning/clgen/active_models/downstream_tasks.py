"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib
import functools
import typing
import tqdm
import multiprocessing
import numpy as np

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import grewe
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import logging as l

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
  def FromTask(cls, task: str, corpus_path: pathlib.Path, random_seed: int) -> "DownstreamTask":
    return TASKS[task](corpus_path, random_seed)

  def __init__(self, name: str, random_seed: int) -> None:
    self.name        = name
    self.random_seed = random_seed
    return

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
  def feature_space(self) -> str:
    return "GreweFeatures"

  def __init__(self, corpus_path: pathlib.Path, random_seed: int) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive", random_seed)
    self.corpus_path    = corpus_path
    self.setup_dataset()
    self.data_generator = data_generator.ListTrainDataloader(self.dataset)

    ## Setup random seed np random stuff
    self.seed_generator = np.random
    self.seed_generator.seed(random_seed)
    self.rand_generators = {}
    max_fval = {
      'comp'      : 300,
      'rational'  : 50,
      'mem'       : 50,
      'localmem'  : 50,
      'coalesced' : 10,
      'atomic'    : 10,
    }
    for fk in grewe.KEYS:
      if fk not in {"F2:coalesced/mem", "F4:comp/mem"}:
        seed = self.seed_generator.randint(0, 2**32-1)
        rgen = np.random
        rgen.seed(seed)
        low_bound = 1 if fk in {"comp", "mem"} else 0
        up_bound  = max_fval[fk]
        self.rand_generators[fk] = lambda: rgen.randint(low_bound, up_bound)
    return

  def __repr__(self) -> str:
    return "GrewePredictive"

  def setup_dataset(self) -> None:
    """
    Fetch data and preprocess into corpus for Grewe's predictive model.
    """
    self.dataset = []
    self.corpus_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus_path)), must_exist = True)
    data    = [x for x in self.corpus_db.get_valid_data()]
    pool = multiprocessing.Pool()
    it = pool.imap_unordered(functools.partial(ExtractorWorker, fspace = "GreweFeatures"), data)
    idx = 0
    try:
      for dp in tqdm.tqdm(it, total = len(data), desc = "Grewe corpus setup", leave = False):
        if dp:
          feats, entry = dp
          self.dataset.append(
            (
              self.InputtoEncodedVector(feats, entry.transferred_bytes, entry.local_size),
              [self.TargetLabeltoID(entry.status)]
            )
          )
          idx += 1
        if idx >= 100:
          break
      pool.close()
    except Exception as e:
      pool.terminate()
      raise e
    return

  def sample_space(self, num_samples: int = 512) -> data_generator.DictPredictionDataloader:
    """
    Go fetch Grewe Predictive model's feature space and randomly return num_samples samples
    to evaluate. The predictive model samples are mapped as a value to the static features
    as a key.
    """
    l.logger().warn("Assuming wgsize (local size) and transferred_bytes is very problematic.")
    samples = []
    for x in range(num_samples):
      fvec = {
        k: self.rand_generators[k]()
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
      samples.append(
        {
          'static_features'  : self.StaticFeatDictToVec(fvec),
          'runtime_features' : [80000, 256],
          'input_ids'        : self.InputtoEncodedVector(fvec, 80000, 256)
        }
      )
    return data_generator.DictPredictionDataloader(samples)

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

TASKS = {
  "GrewePredictive": GrewePredictive,
}
