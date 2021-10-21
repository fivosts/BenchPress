import typing
import pathlib

from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import corpuses

from eupy.native import logger as l

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "clgen_samples_path",
  "",
  "Set path to clgen samples database for evaluation",
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

def initMain(*args, **kwargs):
  raise NotImplementedError
  evaluator = BenchmarkDistance()
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
