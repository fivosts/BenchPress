import typing
import pathlib
import numpy as np

from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.util import monitors

from eupy.native import logger as l

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "clgen_samples_path",
  "",
  "Set path to clgen samples database for evaluation",
)

flags.DEFINE_string(
  "samples_db_path",
  "",
  "Set path to BERT samples database for motivational_example figure",
)

class BaseEvaluator(object):
  """
  Base class for evaluators.
  """
  def __init__(self, sampler: samplers.Sampler):
    self.sampler = sampler
    return

  def eval(self) -> None:
    raise NotImplementedError

class BenchmarkDistance(BaseEvaluator):
  """
  This evaluator is compatible only with active samplers.
  Compares BERT vs CLgen vs Github training data against
  how close their benchmarks are against handwritten benchmarks
  """
  class EvaluatedBenchmark(object):
    """
    Representation of an evaluated benchmark with all its candidates.
    """
    class BenchmarkCandidate(typing.NamedTuple):
      """
      Benchmark candidate
      """
      contents : str
      distance : float
      features : typing.Dict[str, float]
      label    : str

    def __init__(self,
                 target         : pathlib.Path,
                 name           : str,
                 contents       : str,
                 features : typing.Dict[str, float],
                 bert_cands     : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 clgen_cands    : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 github_cands   : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 ) -> None:
      self.target         = target
      self.name           = name
      self.contents       = contents
      self.features = features
      self.bert_cands     = bert_cands
      self.clgen_cands    = clgen_cands
      self.github_cands   = github_cands

  def __init__(self,
               github_corpus : corpuses.Corpus,
               samples_db    : samples_database.SamplesDatabase,
               sampler       : samplers.Sampler
               ) -> None:
    super(BenchmarkDistance, self).__init__(sampler)

    # Target and path to target benchmarks
    self.target      = self.sampler.config.sample_corpus.corpus_config.active.target
    self.target_path = pathlib.Path(feature_sampler.targets[self.target]).resolve()

    # BERT DB setup
    self.bert_db = samples_db

    # clgen DB setup
    self.clgen_samples_path = pathlib.Path(FLAGS.clgen_samples_path)
    if not self.clgen_samples_path.exists():
      raise FileNotFoundError
    self.clgen_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(self.clgen_samples_path)))

    self.github_corpus = github_corpus
    # Feature Space setup
    self.feature_space = self.sampler.config.sample_corpus.corpus_config.active.feature_space

    # self.monitor = monitors.MultiCategoricalDistribution(self.path)
    self.loadBenchmarks()
    return

  def loadBenchmarks(self) -> None:
    """
    Unzip benchmarks zip, iterate, split and collect features for a feature space.
    """
    self.evaluated_benchmarks = []
    kernels = feature_sampler.yield_cl_kernels(self.target_path)
    for p, k, h in kernels:
      features = extractor.ExtractFeatures(k, [self.feature_space], header_file = h, use_aux_headers = False)
      if features[self.feature_space]:
        self.evaluated_benchmarks.append(
          BenchmarkDistance.EvaluatedBenchmark(
              p,
              p.name,
              k,
              features[self.feature_space],
            )
        )
    return

  def eval(self, topK: int) -> None:
    """
    Iterate benchmarks and evaluate datasets efficacy.
    """
    bert_corpus  = [
      (cf, feats[self.feature_space])
      for cf, feats in self.bert_db.get_samples_features
      if self.feature_space in feats
    ]
    clgen_corpus = [
      (cf, feats[self.feature_space])
      for cf, feats in self.clgen_db.get_samples_features
      if self.feature_space in feats
    ]
    git_corpus   = [
      (cf, feats[self.feature_space])
      for cf, feats in self.github_corpus.getFeaturesContents()
      if self.feature_space in feats and feats[self.feature_space]
    ]
    reduced_git_corpus = [
      (cf, feats[self.feature_space])
      for cf, feats in self.github_corpus.getFeaturesContents(sequence_length = self.sampler.sequence_length)
      if self.feature_space in feats and feats[self.feature_space]
    ]
    print(len(clgen_corpus))
    print(len(clgen_corpus))
    print(len(git_corpus))
    print(len(reduced_git_corpus))
    for benchmark in self.evaluated_benchmarks:
      try:
        bc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in bert_corpus ], key = lambda x: x[1])[:topK]
        cc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in clgen_corpus], key = lambda x: x[1])[:topK]
        gc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in git_corpus  ], key = lambda x: x[1])[:topK]
        rgc = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in reduced_git_corpus  ], key = lambda x: x[1])[:topK]
      except KeyError:
        print(git_corpus)

      print(benchmark.name)
      print(benchmark.features)

      print([x[1] for x in bc])
      print([x[1] for x in cc])
      print([x[1] for x in gc])
      print([x[1] for x in rgc])
      input()
    return

def motivational_example_fig():
  """
  Build the plot for paper's motivational example.
  """
  target = "rodinia"
  feature_space = "GreweFeatures"
  benchmarks = []
  kernels = feature_sampler.yield_cl_kernels(pathlib.Path(feature_sampler.targets[target]).resolve())
  for p, k, h in kernels:
    features = extractor.ExtractFeatures(k, [feature_space], header_file = h, use_aux_headers = False)
    if features[feature_space]:
      benchmarks.append(
        BenchmarkDistance.EvaluatedBenchmark(
            p,
            p.name,
            k,
            features[feature_space],
          )
      )
  clgen_samples_path = pathlib.Path(FLAGS.clgen_samples_path).resolve()
  if not clgen_samples_path.exists():
    raise FileNotFoundError
  clgen_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(clgen_samples_path)))
  clgen_corpus = clgen_db.correct_samples

  bert_samples_path = pathlib.Path(FLAGS.samples_db_path).resolve()
  if not bert_samples_path.exists():
    raise FileNotFoundError
  bert_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(bert_samples_path)))
  bert_datapoint = bert_db.get_by_id(50)

  mon = monitors.TSNEMonitor(".", "motivational_example")
  for b in benchmarks:
    mon.register((b.features, "Rodinia Benchmarks", b.name))
  for i in range(20):
    sample = clgen_corpus[np.random.randint(0, len(clgen_corpus))]
    feats = extractor.RawToDictFeats(sample.feature_vector)[feature_space]
    if feature_space in feats and feats[feature_space]:
      mon.register((feats[feature_space], "clgen samples"))

  mon.register((extractor.RawToDictFeats(bert_datapoint[0].feature_vector)[feature_space], "bert sample"))
  mon.plot()
  return

def initMain(*args, **kwargs):
  l.initLogger(name = "evaluators", lvl = 20, mail = (None, 5), colorize = True, step = False)
  motivational_example_fig()
  raise NotImplementedError
  evaluator = BenchmarkDistance()
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
