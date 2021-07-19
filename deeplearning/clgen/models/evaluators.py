import typing
import pathlib

from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import corpuses

from eupy.native import logger as l

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "clgen_samples_path",
  "",
  "Set path to clgen samples database for evaluation".
)

class BaseEvaluator(object):
  def __init__(self, sampler: samplers.Sampler):
    self.sampler = sampler
    self.path = 
    return

  def eval(self) -> None:
    raise NotImplementedError

class BenchmarkDistance(object):
  """
  This evaluator is compatible only with active samplers.
  Compares BERT vs CLgen vs Github training data against
  how close their benchmarks are against handwritten benchmarks
  """

  class BenchmarkCandidate(typing.NamedTuple):
    """
    Benchmark candidate
    """
    contents       : str
    distance       : float
    feature_vector : typing.Dict[str, float]
    label          : str

  class EvaluatedBenchmark(object):
    """
    Representation of an evaluated benchmark with all its candidates.
    """
    def __init__(self,
                 target         : pathlib.Path,
                 name           : str,
                 contents       : str,
                 feature_vector : typing.Dict[str, float],
                 bert_cands     : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 clgen_cands    : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 github_cands   : typing.List[BenchmarkCandidate] = [], # (contents, distance, labels)
                 ) -> None:
      self.target         = target
      self.name           = name
      self.contents       = contents
      self.feature_vector = feature_vector
      self.bert_cands     = bert_cands
      self.clgen_cands    = clgen_cands
      self.github_cands   = github_cands

  def __init__(self, github_corpus: corpuses.Corpus, sampler: samplers.Sampler):
    super(BenchmarkDistance, self).__init__(sampler)
    self.target        = self.sampler.sample_corpus.corpus_config.active.target

    self.target_path   = pathlib.Path(feature_sampler.targets[self.target]).resolve()
    self.clgen_samples_path = pathlib.Path(FLAGS.clgen_samples_path)
    if not self.clgen_samples_path.exists():
      raise FileNotFoundError

    self.feature_space = self.sampler.sample_corpus.corpus_config.active.feature_space
    self.github_corpus = github_corpus

    loadBenchmarks()
    return

  def loadBenchmarks(self) -> None:
    self.evaluated_benchmarks = []
    with feature_sampler.GetContentFileRoot(self.path) as root:
      contentfiles = []
      for file in root.iterdir():
        with open(file, 'r') as inf:
          contentfiles.appendI((file, inf.read()))
    kernels = [(p, k) for k in opencl.ExtractOnlySingleKernels(opencl.InvertKernelSpecifier(cf)) for p, cf in contentfiles]
    for p, k in kernels:
      features = extractor.ExtractFeatures(k, [self.feature_space])
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

  def eval(self) -> None:
    for benchmark in self.evaluated_benchmarks:
      # gather top-K berts, run cldrive for each
      # gather top-K clgen's, run cldrive for each
      
      # gather top-K github's, run cldrive for each
      git = self.github_corpus.getFeaturesContents(self.feature_space)
      # plot stuff + write in json
      pass
    return