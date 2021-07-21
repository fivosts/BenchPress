# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file contains the SampleObserver interface and concrete subclasses."""
import pathlib

from deeplearning.clgen.proto import model_pb2
from absl import flags
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import monitors
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.features import extractor
from labm8.py import fs

FLAGS = flags.FLAGS

class SampleObserver(object):
  """An observer that is notified when new samples are produced.

  During sampling of a model, sample observers are notified for each new
  sample produced. Additionally, sample observers determine when to terminate
  sampling.
  """

  def Specialize(self, model, sampler) -> None:
    """Specialize the sample observer to a model and sampler combination.

    This enables the observer to set state specialized to a specific model and
    sampler. This is guaranteed to be called before OnSample(), and
    sets that the model and sampler for each subsequent call to OnSample(),
    until the next call to Specialize().

    Subclasses do not need to override this method.

    Args:
      model: The model that is being sampled.
      sampler: The sampler that is being used.
    """
    pass

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample notification callback.

    Args:
      sample: The newly created sample message.

    Returns:
      True if sampling should continue, else False. Batching of samples means
      that returning False does not guarantee that sampling will terminate
      immediately, and OnSample() may be called again.
    """
    raise NotImplementedError("abstract class")

  def endSample(self) -> None:
    pass

class MaxSampleCountObserver(SampleObserver):
  """An observer that terminates sampling after a finite number of samples."""

  def __init__(self, min_sample_count: int):
    if min_sample_count <= 0:
      raise ValueError(
        f"min_sample_count must be >= 1. Received: {min_sample_count}"
      )

    self._sample_count = 0
    self._min_sample_count = min_sample_count

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    self._sample_count += 1
    return self._sample_count < self._min_sample_count

class SaveSampleTextObserver(SampleObserver):
  """An observer that creates a file of the sample text for each sample."""

  def __init__(self, path: pathlib.Path):
    self.path = pathlib.Path(path)
    self.path.mkdir(parents=True, exist_ok=True)

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    sample_id = crypto.sha256_str(sample.text)
    path = self.path / f"{sample_id}.txt"
    fs.Write(path, sample.text.encode("utf-8"))
    return True

class PrintSampleObserver(SampleObserver):
  """An observer that prints the text of each sample that is generated."""

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    print(f"=== CLGEN SAMPLE ===\n\n{sample.text}\n")
    return True

class InMemorySampleSaver(SampleObserver):
  """An observer that saves all samples in-memory."""

  def __init__(self):
    self.samples = []

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    self.samples.append(sample)
    return True

class SamplesDatabaseObserver(SampleObserver):
  """A sample observer that imports samples to a database.

  The observer buffers the records that it recieves and commits them to the
  database in batches.
  """

  def __init__(
    self,
    path: pathlib.Path,
    must_exist: bool = False,
    flush_secs: int = 30,
    plot_sample_status = False,
    commit_sample_frequency: int = 1024,
  ):
    self.db = samples_database.SamplesDatabase("sqlite:///{}".format(str(path)), must_exist = must_exist)
    self.sample_id = self.db.count
    self.plot_sample_status = plot_sample_status
    if self.plot_sample_status:
      self.saturation_monitor = monitors.CumulativeHistMonitor(path.parent, "cumulative_sample_count")

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback."""

    with self.db.Session(commit = True) as session:
      db_sample = samples_database.Sample(
        **samples_database.Sample.FromProto(self.sample_id, sample)
      )
      try:
        exists = session.query(samples_database.Sample.sha256).filter_by(sha256 = db_sample.sha256).scalar() is not None
      except sqlalchemy.orm.exc.MultipleResultsFound as e:
        l.getLogger().error("Selected sha256 has been already found more than once.")
        raise e
      if not exists:
        session.add(db_sample)
        self.sample_id += 1
      if self.plot_sample_status:
        self.saturation_monitor.register(self.sample_id)
        self.saturation_monitor.plot()
    return True

  def endSample(self) -> None:
    """Write final summed data about sampling session."""
    # Create feature vector plots
    db_path = pathlib.Path(self.db.url[len("sqlite:///"):]).parent
    # feature_monitor = monitors.CategoricalDistribMonitor(db_path, "samples_feature_vector_distribution")

    feature_monitors = {
      ftype: monitors.CategoricalDistribMonitor(
                        db_path,
                        "{}_distribution".format(ftype)
                      )
      for ftype in extractor.extractors.keys()
    }

    # for sample in self.db.correct_samples:
    #   if sample.feature_vector:
    #     feature_monitor.register({l.split(':')[0:-1]: float(l.split(':')[-1])  for l in sample.feature_vector.split('\n')}) # This used to work only for Grewe. Needs expanding, see lm_data_generator.
    # feature_monitor.plot()

    for sample in self.db.correct_samples:
      if sample.feature_vector:
        features = extractor.RawToDictFeats(sample.feature_vector)
        for ftype, fvector in features.items():
          feature_monitors[ftype].register(fvector)

    for mon in feature_monitors.values():
      mon,plot()

    with self.db.Session() as session:
      compiled_count = session.query(samples_database.Sample.compile_status).filter_by(compile_status = 1).count()
    try:
      r = [
        'compilation rate: {}'.format(compiled_count / self.sample_id),
        'total compilable samples: {}'.format(compiled_count),
        'average feature vector: \n{}'.format(feature_monitor.getStrData())
      ]
    except ZeroDivisionError:
      r = [
        'compilation rate: +/-inf',
        'total compilable samples: {}'.format(compiled_count),
        'average feature vector: \n{}'.format(feature_monitor.getStrData())
      ]
    with self.db.Session(commit = True) as session:
      exists  = session.query(samples_database.SampleResults.key).filter_by(key = "meta").scalar() is not None
      if exists:
        entry = session.query(samples_database.SampleResults    ).filter_by(key = "meta").first()
        entry.results = "\n".join(r)
      else:
        session.add(samples_database.SampleResults(key = "meta", results = "\n".join(r)))
    return

class LegacySampleCacheObserver(SampleObserver):
  """Backwards compatability implementation of the old sample caching behavior.

  In previous versions of CLgen, model sampling would silently (and always)
  create sample protobufs in the sampler cache, located at:

    CLGEN_CACHE/models/MODEL/samples/SAMPLER

  This sample observer provides equivalent behavior.
  """

  def __init__(self):
    self.cache_path = None

  def Specialize(self, model, sampler) -> None:
    """Specialize observer to a model and sampler combination."""
    self.cache_path = model.SamplerCache(sampler)
    self.cache_path.mkdir(exist_ok=True)

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    sample_id = crypto.sha256_str(sample.text)
    sample_path = self.cache_path / f"{sample_id}.pbtxt"
    pbutil.ToFile(sample, sample_path)
    return True
