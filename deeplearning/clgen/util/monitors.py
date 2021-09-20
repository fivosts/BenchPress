"""Statistical distributions used for sampling"""
import pathlib
import pickle
import typing
import numpy as np
import sklearn.manifold

from deeplearning.clgen.util import plotter

from eupy.native import logger as l

class Monitor():
  def __init__(self,
               cache_path : typing.Union[pathlib.Path, str],
               set_name   : str
               ):
    self.cache_path     = cache_path if isinstance(cache_path, pathlib.Path) else pathlib.Path(cache_path)
    self.set_name       = set_name
    return

  def saveCheckpoint(self):
    with open(self.cache_path / "{}_state.pkl".format(self.set_name), 'wb') as outf:
      pickle.dump(self, outf)
    return

  @classmethod
  def loadCheckpoint(cls, cache_path, set_name):
    if (cache_path / "{}_state.pkl".format(set_name)).exists():
      with open(cache_path / "{}_state.pkl".format(set_name), 'rb') as infile:
        obj = pickle.load(infile)
    else:
      obj = cls(cache_path, set_name)
    return obj

  def getData(self):
    raise NotImplementedError("Abstract Class")

  def getStrData(self):
    raise NotImplementedError("Abstract Class")

  def register(self, actual_sample):
    raise NotImplementedError("Abstract Class")

  def sample(self):
    raise NotImplementedError("Abstract Class")

  def plot(self):
    raise NotImplementedError("Abstract Class")

class FrequencyMonitor(Monitor):
  """
  Keeps monitor of the occured frequency of a specific key.
  Key is provided through `actual_sample` in register method.
  Its frequency is incremented by one.

  Bar plots num of occurences VS keys.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name  : str,
               ):
    super(FrequencyMonitor, self).__init__(cache_path, set_name)
    self.sample_counter = {}
    return

  def getData(self) -> typing.List[typing.Tuple[typing.Union[int, str, float], int]]:
    return sorted(self.sample_counter.items(), key = lambda x: x[0])

  def getStrData(self) -> str:
    return "\n".join(
      ["{}:{}".format(k, v) for (k, v) in self.getData()]
    )

  def register(self, actual_sample: typing.Union[int, str, list]) -> None:
    if isinstance(actual_sample, list):
      for s in actual_sample:
        self.register(s)
    else:
      if actual_sample not in self.sample_counter:
        self.sample_counter[actual_sample] =  1
      else:
        self.sample_counter[actual_sample] += 1
    return

  def plot(self) -> None:
    """Plot bars of number of occurences."""
    sorted_dict = sorted(self.sample_counter.items(), key = lambda x: x[0])
    plotter.FrequencyBars(
      x = [x for (x, _) in sorted_dict],
      y = [y for (_, y) in sorted_dict],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )
    return

class NormalizedFrequencyMonitor(FrequencyMonitor):
  """
  Identical to FrequencyMonitor but normalizes absolute values
  of bars with respect to total occurrences.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name  : str,
               ):
    super(NormalizedFrequencyMonitor, self).__init__(cache_path, set_name)
    return

  def plot(self) -> None:
    """Plot bars of number of occurences."""
    total = sum(self.sample_counter.values())
    sorted_dict = sorted(self.sample_counter.items(), key = lambda x: x[0])
    plotter.FrequencyBars(
      x = [x for (x, _) in sorted_dict],
      y = [y / total for (_, y) in sorted_dict],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )
    return

class CumulativeHistMonitor(Monitor):
  """
  Keeps monitor of the occured frequency of a specific key.
  Key is provided through `actual_sample` in register method.
  Its frequency is incremented by one.

  Bar plots num of occurences VS keys.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name  : str,
               ):
    super(CumulativeHistMonitor, self).__init__(cache_path, set_name)
    self.sample_counter = {}
    return

  def getData(self) -> typing.List[typing.Tuple[typing.Union[int, str, float], int]]:
    return sorted(self.sample_counter.items(), key = lambda x: x[0])

  def getStrData(self) -> str:
    return "\n".join(
      ["{}:{}".format(k, v) for (k, v) in self.getData()]
    )

  def register(self, actual_sample: typing.Union[list, int, float]) -> None:
    if isinstance(actual_sample, list):
      for s in actual_sample:
        self.register(s)
    else:
      if actual_sample not in self.sample_counter:
        self.sample_counter[actual_sample] =  1
      else:
        self.sample_counter[actual_sample] += 1
    return

  def plot(self) -> None:
    """Plot bars of number of occurences."""
    sorted_dict = self.getData()
    plotter.CumulativeHistogram(
      x = [x for (x, _) in sorted_dict],
      y = [y for (_, y) in sorted_dict],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )
    return

class HistoryMonitor(Monitor):
  """
  Monitors values in an ordered timeline
  Plots a line of values against timesteps.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(HistoryMonitor, self).__init__(cache_path, set_name)
    self.sample_list = []
    return

  def getData(self) -> typing.List[typing.Union[int, float]]:
    return self.sample_list

  def getStrData(self) -> str:
    return ",".join(
      [str(v) for v in self.getData()]
    )

  def register(self, actual_sample: typing.Union[int, float]) -> None:
    self.sample_list.append(float(actual_sample))
    return

  def plot(self) -> None:
    """Plot line over timescale"""
    plotter.SingleScatterLine(
      x = np.arange(len(self.sample_list)),
      y = self.sample_list,
      title = self.set_name,
      x_name = "",
      y_name = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path,
    )
    return

class CategoricalHistoryMonitor(Monitor):
  """
  Scatter line of one datapoint per category.
  Useful to track average value per category.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(CategoricalHistoryMonitor, self).__init__(cache_path, set_name)
    self.sample_dict = {}
    return

  def getData(self) -> typing.List[typing.Tuple[typing.Union[int, str, float], int]]:
    return sorted(self.sample_dict.items(), key = lambda x: x[0])

  def getStrData(self) -> str:
    return "\n".join(
      ["{}:{}".format(k, v) for (k, v) in self.getData()]
    )

  def register(self, actual_sample: typing.Tuple[typing.Any, typing.Any]) -> None:
    key, value = actual_sample
    self.sample_dict[key] = value
    return

  def plot(self) -> None:
    """Plot line over timescale"""
    sorted_dict = self.getData()
    plotter.SingleScatterLine(
      x = [x for (x, _) in sorted_dict],
      y = [y for (_, y) in sorted_dict],
      title = self.set_name,
      x_name = "",
      y_name = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path,
    )
    return

class CategoricalDistribMonitor(Monitor):
  """
  Monitors values in an ordered timeline
  Plots a line of values against timesteps.
  X-Axis can be regulated when registering a new value.
  The new value per x-element is always going to be the minimum seen.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(CategoricalDistribMonitor, self).__init__(cache_path, set_name)
    self.sample_dict = {}
    return

  def getData(self) -> typing.List[typing.Tuple[typing.Union[int, str, float], int]]:
    return sorted(self.sample_dict.items(), key = lambda x: x[0])

  def getStrData(self) -> str:
    return "\n".join(
      ["{}:{}".format(k, sum(v) / len(v)) for (k, v) in self.getData()]
    )

  def register(self, actual_sample: typing.Dict[str, float]) -> None:
    for k, v in actual_sample.items():
      if isinstance(v, list):
        val = v
      else:
        val = [v]
      if k in self.sample_dict:
        self.sample_dict[k] += val
      else:
        self.sample_dict[k] = val
    return

  def plot(self) -> None:
    """Plot line over timescale"""
    sorted_dict = self.getData()
    plotter.CategoricalViolin(
      x = [k for (k, _) in sorted_dict],
      y = [v for (_, v) in sorted_dict],
      title = self.set_name,
      x_name = "",
      plot_name = self.set_name,
      path = self.cache_path,
    )
    return

class FeatureMonitor(Monitor):
  """
  Produces a bar chart of averaged features.
  Yes, features are averaged. It is not a cumulative representation.
  """
  def __init__(self,
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(FeatureMonitor, self).__init__(cache_path, set_name)
    self.features = {}
    self.instance_counter = 0
    return

  def getData(self) -> typing.Dict[str, float]:
    return {k: v / self.instance_counter for k, v in self.features.items()}

  def getStrData(self) -> str:
    return "\n".join(
      ["{}:{}".format(k, v) for k, v in self.getData().items()]
    )

  def register(self, actual_sample: typing.Dict[str, float]) -> None:
    """actual sample is a dict of features to their values."""
    if not isinstance(actual_sample, dict):
      raise TypeError("Feature sample must be dictionary of string features to float values. Received: {}".format(actual_sample))

    self.instance_counter += 1
    for k, v in actual_sample.items():
      if k not in self.features:
        self.features[k] = v
      else:
        self.features[k] += v
    return

  def plot(self) -> None:
    """Plot averaged Bar chart"""
    plotter.FrequencyBars(
      x = [k for k in self.features.keys()],
      y = [v / self.instance_counter for v in self.features.values()],
      title     = self.set_name,
      x_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )

class TSNEMonitor(Monitor):
  """
  Keeps track of feature vectors of various groups in a given feature space.
  Performs t-SNE algorithm to reduce dimensionality to 2 and plots groupped scatterplot.
  """
  def __init__(self, 
               cache_path: typing.Union[pathlib.Path, str],
               set_name: str,
               ):
    super(TSNEMonitor, self).__init__(cache_path, set_name)
    self.features     = []
    self.features_set = set()
    self.groups       = []
    self.groups_set   = set()
    self.names        = []
    return

  def getData(self) -> None:
    raise NotImplementedError

  def getStrData(self) -> None:
    raise NotImplementedError

  def register(self, actual_sample: typing.Tuple[typing.Dict[str, int], str, typing.Optional[str]]) -> None:
    """
    A registered sample must contain:
    1. A feature vector.
    2. The group it belongs to.
    3. (Optional) The name of the datapoint.

    Feature vectors stored are unique.
    """
    feats, group = actual_sample[0], actual_sample[1]
    name = actual_sample[2] if len(actual_sample) == 3 else ""
    feats_list = list(feats.values())
    if str(feats_list) not in self.features_set:
      self.features.append(feats_list)
      self.features_set.add(str(feats_list))
      self.groups.append(group)
      self.groups_set.add(group)
      self.names.append(name)
    elif name != "":
      for idx, f in enumerate(self.features):
        if feats_list == f:
          self.names[idx] += ",{}".format(name)
    return

  def plot(self) -> None:
    """
    Plot groupped scatter graph.
    """
    tsne = sklearn.manifold.TSNE()
    embeddings = tsne.fit_transform(np.array(self.features))
    groupped_data = {}
    for points, group, name in zip(embeddings, self.groups, self.names):
      if group in groupped_data:
        groupped_data[group]['data'].append(points)
        groupped_data[group]['names'].append(name)
      else:
        groupped_data[group] = {
          'data': [points],
          'names': [name],
        }
    plotter.GroupScatterPlot(
      groups    = groupped_data,
      title     = self.set_name,
      x_name    = self.set_name,
      y_name    = self.set_name,
      plot_name = self.set_name,
      path = self.cache_path
    )