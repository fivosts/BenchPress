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
from deeplearning.clgen.util import plotter

from eupy.native import logger as l

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "clgen_samples_path",
  None,
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
    if not FLAGS.clgen_samples_path:
      raise ValueError("clgen_samples_path has not been set for evaluation")
    self.clgen_samples_path = pathlib.Path(FLAGS.clgen_samples_path).resolve()
    if not self.clgen_samples_path.exists():
      raise FileNotFoundError(str(self.clgen_samples_path))
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
    l.getLogger().info("Loaded {} benchmarks".format(len(self.evaluated_benchmarks)))
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
    self.avg_score_per_target(bert_corpus, clgen_corpus, git_corpus, reduced_git_corpus, topK)

    outfile = pathlib.Path("./results.out").resolve()
    with open(outfile, 'w') as outf:
      for benchmark in self.evaluated_benchmarks:
        try:
          bc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in bert_corpus ], key = lambda x: x[1])[:topK]
          cc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in clgen_corpus], key = lambda x: x[1])[:topK]
          gc  = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in git_corpus  ], key = lambda x: x[1])[:topK]
          rgc = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in reduced_git_corpus  ], key = lambda x: x[1])[:topK]
        except KeyError:
          print(git_corpus)

        if rgc[0][1] > 0:
          outf.write("###############################\n")
          outf.write("{}\n".format(benchmark.name))
          outf.write("Target features: {}\n".format(benchmark.features))

          outf.write("BenchPress: {}\n".format([x[1] for x in bc]))

          outf.write("CLgen: {}\n".format([x[1] for x in cc]))
          outf.write("Github: {}\n".format([x[1] for x in gc]))
          outf.write("Github <= 768 tokens: {}\n".format([x[1] for x in rgc]))

          outf.write("\n\nTop-5 functions BenchPress's functions: \n")
          for x in bc:
            outf.write("{}\n\n".format(x[0]))
          outf.write("###############################\n\n")

    return

  def avg_score_per_target(self, bert, clgen, git, reduced_git, topK):
    groups = {}
    names = {}
    final_benchmarks = []

    for benchmark in self.evaluated_benchmarks:
      rgc = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in reduced_git], key = lambda x: x[1])[:topK]
      if rgc[0][1] > 0:
        final_benchmarks.append(benchmark)

    for benchmark in final_benchmarks:
      if benchmark.name not in names:
        names[benchmark.name] = [0, 0]
      else:
        names[benchmark.name][1] += 1

    l.getLogger().info("Plotting {} benchmarks".format(len(final_benchmarks)))

    for dsetname, dset in [("BERT", bert), ("CLgen", clgen), ("GitHub", git), ("Reduced GitHub", reduced_git)]:
      groups[dsetname] = ([], [])
      for benchmark in final_benchmarks:
        if names[benchmark.name][1] == 0:
          bn = benchmark.name
        else:
          names[benchmark.name][0] += 1
          bn = "{}-{}".format(benchmark.name, names[benchmark.name][0])
        dst_list = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in dset], key = lambda x: x[1])[:topK]
        groups[dsetname][0].append(bn)
        # groups[dsetname][1].append(sum([x[1] for x in dst_list]) / len(dst_list))
        groups[dsetname][1].append(min([x[1] for x in dst_list]))
      for benchmark in final_benchmarks:
        names[benchmark.name][0] = 0
    plotter.GrouppedBars(
      groups = groups,
      title  = "Grewe Features",
      x_name = "",
      plot_name = "avg_dist",
      path = pathlib.Path(".").resolve(),
    )
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
  clgen_corpus = [s for s in clgen_db.correct_samples]

  bert_samples_path = pathlib.Path(FLAGS.samples_db_path).resolve()
  if not bert_samples_path.exists():
    raise FileNotFoundError
  bert_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(bert_samples_path)))
  bert_datapoints = bert_db.get_by_ids([143826, 146576, 144315])
  data = [s for s in bert_db.correct_samples]
  # for x in range(20):
  #   s = data[np.random.randint(0, len(data))]
  #   feats = extractor.RawToDictFeats(s.feature_vector)
  #   if feature_space in feats and feats[feature_space]:
  #     bert_datapoints.append(s)
  #   else:
  #     s.feature_vector = extractor.ExtractRawFeatures(s.text)

  groups = {
    "Rodinia Benchmarks": {'data': [], 'names': []},
    "CLgen samples": {'data': [], 'names': []},
    "BenchPress Examples": {'data': [], 'names': []},
  }

  mon = monitors.TSNEMonitor(".", "motivational_example")
  for b in benchmarks:
    mon.register((b.features, "Rodinia Benchmarks", b.name))
    groups["Rodinia Benchmarks"]['data'].append([b.features['comp'], b.features['mem']])
    groups["Rodinia Benchmarks"]['names'].append(b.name)
  for i in range(30):
    sample = clgen_corpus[np.random.randint(0, len(clgen_corpus))]
    feats = extractor.RawToDictFeats(sample.feature_vector)
    if feature_space in feats and feats[feature_space]:
      mon.register((feats[feature_space], "clgen samples"))
      groups["CLgen samples"]['data'].append([feats[feature_space]['comp'], feats[feature_space]['mem']])
      groups["CLgen samples"]['names'].append("")

  for s in bert_datapoints:
    feats = extractor.RawToDictFeats(s.feature_vector)[feature_space]
    # for k in feats.keys():
    #   feats[k] += 100
    mon.register((feats, "bert sample"))
    groups["BenchPress Examples"]["data"].append([feats['comp'], feats['mem']])
  mon.plot()
  plotter.GroupScatterPlot(
    groups = groups,
    title = "",
    x_name = "# Computational Instructions",
    y_name = "# Memory Instructions",
    plot_name = "motivational_example_compvsmem",
    marker_style = [
      dict(color = 'darkslateblue', size = 10, symbol = "diamond-open", line = dict(width = 4)),
      dict(color = 'goldenrod', size = 10, symbol = "circle"),
      dict(color = 'firebrick', size = 14, symbol = "cross"),
    ],
    path = pathlib.Path(".").resolve(),
  )
  return

def benchpress_vs_clgen_fig():

  clgen_samples_path = pathlib.Path(FLAGS.clgen_samples_path).resolve()
  if not clgen_samples_path.exists():
    raise FileNotFoundError
  clgen_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(clgen_samples_path)))
  clgen_ntoks = clgen_db.get_compilable_num_tokens

  clgen_num_insts = [x[1]["InstCountFeatures"]["TotalInsts"] for x in clgen_db.get_samples_features if "InstCountFeatures" in x[1]]

  bert_samples_path = pathlib.Path(FLAGS.samples_db_path).resolve()
  if not bert_samples_path.exists():
    raise FileNotFoundError
  bert_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(bert_samples_path)))
  data = bert_db.get_compilable_num_tokens

  bert_num_insts = [x[1]["InstCountFeatures"]["TotalInsts"] for x in bert_db.get_samples_features if "InstCountFeatures" in x[1]]

  plotter.RelativeDistribution(
    x = ["BenchPress", "CLgen"],
    y = [data, clgen_ntoks],
    title = "",
    x_name = "# Tokens",
    plot_name = "token_relative_distribution",
    path = pathlib.Path(".").resolve()
  )

  plotter.RelativeDistribution(
    x = ["BenchPress", "CLgen"],
    y = [bert_num_insts, clgen_num_insts],
    title = "",
    x_name = "# LLVM IR Instructions (-O1)",
    plot_name = "numinst_relative_distribution",
    path = pathlib.Path(".").resolve()
  )

def initMain(*args, **kwargs):
  l.initLogger(name = "evaluators", lvl = 20, mail = (None, 5), colorize = True, step = False)
  # motivational_example_fig()
  benchpress_vs_clgen_fig()
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
