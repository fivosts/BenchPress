import typing
import pathlib
import sklearn
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

from deeplearning.clgen.proto import evaluator_pb2
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.util import monitors
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import pbutil

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

flags.DEFINE_string(
  "evaluator_config",
  "",
  "Set path to evaluator config file",
)

class DBGroup(object):
  """
  Class representation of a group of databases evaluated.
  """
  @property
  def data(self):
    """
    Get concatenated data of all databases.
    """
    if self.data:
      return self.data
    else:
      self.data = []
      for db in self.databases:
        self.data += db.get_data

  @property
  def features(self):
    raise NotImplementedError
    return None
  
  def __init__(self, group_name: str, group_type: str, databases: typing.List[pathlib.Path]):
    self.group_name = group_name
    self.group_type = {
      "SamplesDatabase"    : samples_database.SamplesDatabase,
      "ActiveFeedDatabase" : active_feed_database.ActiveFeedDatabase,
    }[group_type]
    self.databases = [self.group_type("sqlite:///{}".format(pathlib.Path(p).resolve())) for p in databases]


class Benchmark(typing.NamedTuple):
  path     : pathlib.Path
  name     : str
  contents : str
  features : typing.Dict[str, float]

class TargetBenchmarks(object):
  """
  Class representation of target benchmarks.
  """
  def __init__(self, target: str):
    self.target        = target
    self.benchmark_cfs = feature_sampler.yield_cl_kernels(pathlib.Path(feature_sampler.targets[self.target]).resolve())
    self.benchmarks    = {ext: None for ext in extractor.extractors.keys()}

  def get_features(self, feature_space: str):
    """
    Get or set and get benchmarks with their features for a feature space.
    """
    if self.features[feature_space]:
      return self.features[feature_space]
    else:
      for p, k, h in self.benchmark_cfs:
        features = extractor.ExtractFeatures(k, [feature_space], header_file = h, use_aux_headers = False)
        if features[feature_space]:
          self.benchmarks[feature_space].append(
            BenchmarkDistance.EvaluatedBenchmark(
                p,
                p.name,
                k,
                features[feature_space],
              )
          )

def AssertIfValid(config: evaluator_pb2.Evaluation):
  """
  Parse config file and check for validity.
  """

  for ev in config.evaluator:
    if ev.HasField("k_average_score"):
      for dbs in ev.k_average_score.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
      pbutil.AssertFieldConstraint(
        ev.k_average_score,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.k_average_score.target),
      )
      pbutil.AssertFieldIsSet(ev.k_average_score, "feature_space")
      pbutil.AssertFieldConstraint(
        ev.k_average_score,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
    elif ev.HasField("min_score"):
      for dbs in ev.min_score.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
      pbutil.AssertFieldConstraint(
        ev.min_score,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.min_score.target),
      )
      pbutil.AssertFieldIsSet(ev.min_score, "feature_space")
    elif ev.HasField("analyze_target"):
      for dbs in ev.analyze_target.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
      pbutil.AssertFieldIsSet(ev.analyze_target, "tokenizer")
      if not pathlib.Path(ev.analyze_target.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(ev.analyze_target.tokenizer).resolve())
      for target in ev.analyze_target.targets:
        assert target in feature_sampler.targets, target
    elif ev.HasField("log_file"):
      for dbs in ev.log_file.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
    elif ev.HasField("comp_mem_grewe"):
      for dbs in ev.comp_mem_grewe.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
      pbutil.AssertFieldConstraint(
        ev.comp_mem_grewe,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.comp_mem_grewe.target),
      )
  return config

def ConfigFromFlags() -> evaluator_pb2.Evaluation:
  """
  Parse evaluator config path and return config.
  """
  config_path = pathlib.Path(FLAGS.evaluator_config)
  if not config_path.is_file():
    raise FileNotFoundError (f"Evaluation --evaluator_config file not found: '{config_path}'")
  config = pbutil.FromFile(config_path, evaluator_pb2.Evaluation())
  return AssertIfValid(config)

def LogFile(**kwargs):
  """
  Write benchmarks  and target stats in log file.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')

  return

def KAverageScore(**kwargs):
  """
  Compare the average of top-K closest per target benchmark
  for all different database groups.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')
  feature_space = kwargs.get('feature_space')
  top_k         = kwargs.get('top_k')

  return

def MinScore(**kwargs):
  """
  Compare the closest sample per target benchmark
  for all different database groups.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')
  feature_space = kwargs.get('feature_space')

  return

def AnalyzeTarget(**kwargs):
  """
  Analyze requested target benchmark suites.
  """
  targets   = kwargs.get('targets')
  tokenizer = kwargs.get('tokenizer')

  return

def CompMemGrewe(**kwargs):
  """
  Compare Computation vs Memory instructions for each database group
  and target benchmarks.
  """
  db_groups     = kwargs.get('db_groups')
  target        = kwargs.get('targets')

  return

def main(config: evaluator_pb2.Evaluation):
  """
  Run the evaluators iteratively.
  """
  evaluation_map = {
    evaluator_pb2.LogFile       : LogFile,
    evaluator_pb2.KAverageScore : KAverageScore,
    evaluator_pb2.MinScore      : MinScore,
    evaluator_pb2.AnalyzeTarget : AnalyzeTarget,
    evaluator_pb2.CompMemGrewe  : CompMemGrewe,
  }
  db_cache       = {}
  target_cache   = {}
  tokenizer      = None
  feature_spaces = []
  for ev in config.evaluator:
    kw_args = {
      "db_groups"      : [],
      "targets"        : None,
      "feature_space"  : None,
      "tokenizer"      : None,
      "top_k"          : None,
    }
    if ev.HasField("k_average_score"):
      sev = ev.k_average_score
      kw_args['top_k'] = sev.top_k
    elif ev.HasField("min_score"):
      sev = ev.min_score
    elif ev.HasField("analyze_target"):
      sev = ev.analyze_target
      kw_args['tokenizer'] = tokenizers.FromFile(pathlib.Path(sev.tokenizer).resolve())
    elif ev.HasField("log_file"):
      sev = ev.log_file
    elif ev.HasField("comp_mem_grewe"):
      sev = ev.comp_mem_grewe
    else:
      raise NotImplementedError(ev)

    # Gather database groups and cache them
    for dbs in sev.db_group:
      key = dbs.group_name + ''.join(dbs.database)
      if key not in db_cache:
        db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database)
      kw_args['db_groups'].append(db_cache[key])
    # Gather target benchmarks and cache them
    if sev.HasField("target"):
      if isinstance(sev.target, list):
        kw_args["targets"] = []
        for t in sev.target:
          if t not in target_cache:
            target_cache[t] = TargetBenchmarks(t)
          kw_args["targets"].append(target_cache[t])
      else:
        if sev.target not in target_cache:
          target_cache[sev.target] = TargetBenchmarks(sev.target)
          kw_args["targets"] = target_cache[sev.target]
    # Gather feature spaces if applicable.
    if sev.HasField("feature_space"):
      kw_args['feature_space'] = sev.feature_space

    evaluation_map[type(sev)](**kw_args)
  return

def initMain(*args, **kwargs):
  l.initLogger(name = "evaluators", lvl = 20, mail = (None, 5), colorize = True, step = False)
  config = ConfigFromFlags()
  main(config)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)

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
                 features       : typing.Dict[str, float],
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
               tokenizer,
               github_corpus : corpuses.Corpus,
               samples_db    : samples_database.SamplesDatabase,
               sampler       : samplers.Sampler
               ) -> None:
    super(BenchmarkDistance, self).__init__(sampler)

    # Target and path to target benchmarks
    self.tokenizer   = tokenizer
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

    outfile = pathlib.Path("./results_{}.out".format(self.feature_space)).resolve()
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

          outf.write("BenchPress: {}, lengths: {}\n".format([x[1] for x in bc], [len(self.tokenizer.TokenizeString(x[0])) for x in bc]))

          outf.write("CLgen: {}, lengths: {}\n".format([x[1] for x in cc], [len(self.tokenizer.TokenizeString(x[0])) for x in cc]))
          outf.write("Github: {}, lengths: {}\n".format([x[1] for x in gc], [len(self.tokenizer.TokenizeString(x[0])) for x in gc]))
          outf.write("Github <= 768 tokens: {}, lengths: {}\n".format([x[1] for x in rgc], [len(self.tokenizer.TokenizeString(x[0])) for x in rgc]))

          outf.write("\n\nTop-5 functions BenchPress's functions: \n")
          for x in bc:
            outf.write("{}\n\n".format(x[0]))
            outf.write("Number of tokens: {}\nNumber of LLVM IR Instructions: {}\n\n".format(len(self.tokenizer.TokenizeString(x[0])), extractor.ExtractFeatures(x[0], ["InstCountFeatures"])["InstCountFeatures"]["TotalInsts"]))
          outf.write("###############################\n\n")

    # self.avg_score_per_target(bert_corpus, clgen_corpus, git_corpus, reduced_git_corpus, topK)
    # self.benchmark_stats(reduced_git_corpus)
    return

  def avg_score_per_target(self, bert, clgen, git, reduced_git, topK):

    # I have skipped the following autophase and instcount benchmarks
    # because my experiments weren't finished for them.
    # Then, I change the names of grewe's benchmarks for those in common with auto and inst.
    autophase_excluded_benchmarks = {
      "particle_naive.cl",
      "particle_single.cl-1",
      "particle_single.cl-2",
      "particle_single.cl-3",
      "particle_single.cl-4",
      "particle_double.cl-1",
      "particle_double.cl-2",
      "particle_double.cl-3",
      "particle_double.cl-4",
      "mergesort.cl-2",
      "nw.cl-1",
      "nw.cl-2",
      "track_ellipse_kernel.cl",
      "track_ellipse_kernel_opt.cl",
    }

    instcount_excluded_benchmarks = {
      "particle_naive.cl",
      "particle_single.cl-1",
      "particle_single.cl-2",
      "particle_single.cl-3",
      "particle_single.cl-4",
      "particle_double.cl-3",
      "particle_double.cl-4",
      "track_ellipse_kernel.cl",
      "track_ellipse_kernel_opt.cl",
      "gaussianElim_kernels.cl-1",
      "gaussianElim_kernels.cl-2",
      "nearestNeighbor_kernel.cl-1",
      "find_ellipse_kernel.cl-1",
      "Kernels.cl-5",
      "Kernels.cl-9",
    }

    grewe_transformations = {
      "Kernels.cl-1": "Kernels.cl-2",
      "Kernels.cl-2": "Kernels.cl-3",
      "particle_single.cl-1": "particle_single.cl-2",
      "particle_single.cl-2": "particle_single.cl-4",
      "particle_double.cl-1": "particle_double.cl-2",
      "particle_double.cl-2": "particle_double.cl-4",
      "bucketsort_kernels.cl": "bucketsort_kernels.cl-1",
      "kernel_gpu_opencl.cl-2": "kernel_gpu_opencl.cl-3",
      "kernel_gpu_opencl.cl-3": "kernel_gpu_opencl.cl-4",
    }

    groups = {}
    names = {}
    final_benchmarks = []

    bert_samples_path = pathlib.Path(FLAGS.samples_db_path).resolve()
    if not bert_samples_path.exists():
      raise FileNotFoundError
    fixed_bert_db = samples_database.SamplesDatabase("sqlite:///{}".format(str(bert_samples_path)))
    extra_bert_corpus  = [
      (cf, feats[self.feature_space])
      for cf, feats in fixed_bert_db.get_samples_features
      if self.feature_space in feats
    ]

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

    # for dsetname, dset in [("BenchPress", bert), ("CLgen", clgen), ("GitHub", git), ("GitHub-768", reduced_git)]:
    for dsetname, dset in [("BenchPress", bert + extra_bert_corpus), ("CLgen", clgen), ("GitHub", git), ("GitHub-768", reduced_git)]:
      groups[dsetname] = ([], [])
      for benchmark in final_benchmarks:
        if names[benchmark.name][1] == 0:
          bn = benchmark.name
        else:
          names[benchmark.name][0] += 1
          bn = "{}-{}".format(benchmark.name, names[benchmark.name][0])
        if (self.feature_space == "AutophaseFeatures" and bn in autophase_excluded_benchmarks) or (self.feature_space == "InstCountFeatures" and bn in instcount_excluded_benchmarks):
          continue
        if self.feature_space == "GreweFeatures" and bn in grewe_transformations:
          bn = grewe_transformations[bn]
    
        baseline_distance = math.sqrt(sum([x**2 for x in benchmark.features.values()]))

        dst_list = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features, self.feature_space)) for cf, fts in dset], key = lambda x: x[1])[:topK]
        groups[dsetname][0].append(bn)
        # groups[dsetname][1].append(sum([x[1] for x in dst_list]) / len(dst_list))
        # groups[dsetname][1].append(min([x[1] for x in dst_list]))

        ## Toggle this for normalization from baseline distance.
        groups[dsetname][1].append(100 * ((baseline_distance - dst_list[0][1]) / baseline_distance))
      for benchmark in final_benchmarks:
        names[benchmark.name][0] = 0
    plotter.GrouppedBars(
      groups = groups,
      plot_name = "avg_dist_{}".format(self.feature_space.replace("Features", " Features")),
      path = pathlib.Path(".").resolve(),
      title  = "{}".format(self.feature_space.replace("Features", " Features")),
    )
    return

  def benchmark_stats(self, reduced_git):
    groups = {}
    names = {}
    final_benchmarks = []

    benchmarks = []
    kernels = feature_sampler.yield_cl_kernels(self.target_path)
    for p, k, h in kernels:
      features = extractor.ExtractFeatures(k, header_file = h, use_aux_headers = False)
      if features:
        benchmarks.append(
          BenchmarkDistance.EvaluatedBenchmark(
              p,
              p.name,
              k,
              features,
            )
        )

    for benchmark in benchmarks:
      rgc = sorted([(cf, feature_sampler.calculate_distance(fts, benchmark.features[self.feature_space], self.feature_space)) for cf, fts in reduced_git], key = lambda x: x[1])[0]
      if rgc[1] > 0:
        final_benchmarks.append(benchmark)

    for benchmark in final_benchmarks:
      if benchmark.name not in names:
        names[benchmark.name] = [0, 0]
      else:
        names[benchmark.name][1] += 1

    l.getLogger().info("Analyzing {} benchmarks".format(len(final_benchmarks)))

    for benchmark in final_benchmarks:
      if names[benchmark.name][1] == 0:
        bn = benchmark.name 
      else:
        names[benchmark.name][0] += 1
        bn = "{}-{}".format(benchmark.name, names[benchmark.name][0])
      src = opencl.SanitizeKernelPrototype(opencl.NormalizeIdentifiers(benchmark.contents))
      print("########################")
      print(bn)
      print("Num of tokens", len(self.tokenizer.TokenizeString(src)))
      try:
        print("Number of IR instructions", benchmark.features["InstCountFeatures"]["TotalInsts"])
      except Exception:
        print("Number of IR instructions", "-1")
    return

def kmeans_datasets(bert_db, clgen_db):

  scaler = sklearn.preprocessing.StandardScaler()
  for fspace in ["AutophaseFeatures", "InstCountFeatures", "GreweFeatures"]:

    bert_ds = [x for _, x in bert_db.get_samples_features if fspace in x]
    clgen_ds = [x for _, x in clgen_db.get_samples_features if fspace in x]

    data = []
    for x in bert_ds + clgen_ds:
      vals = list(x[fspace].values())
      if vals:
        data.append([float(y) for y in vals])

    bert_scaled = scaler.fit_transform(data)
    # bert_scaled = data
    bert_reduced = PCA(2).fit_transform(data)
    # clgen_scaled = scaler.fit_transform([[float(y) for y in x[1].values()] for x in clgen])
    # git_scaled = scaler.fit_transform([[float(y) for y in x[1].values()] for x in git])
    # reduced_git_scaled = scaler.fit_transform([[float(y) for y in x[1].values()] for x in reduced_git])

    kmeans = KMeans(
      init = "random",
      n_clusters = 4,
      n_init = 20,
      max_iter = 300,
      random_state = 42,
    )
    kmeans.fit(bert_reduced)
    groups = {
      "Benchpress": {"names": [], "data": bert_reduced[:len(bert_ds)]},
      "CLgen": {"names": [], "data": bert_reduced[len(bert_ds):len(bert_ds) + len(clgen_ds)]},
      # "GitHub": {"names": [], "data": bert_reduced[len(bert) + len(clgen):]},
    }
    plotter.GroupScatterPlot(
      groups = groups,
      title = "PCA-2 {}".format(fspace.replace("Features", " Features")),
      plot_name = "pc2_{}_bpclgen".format(fspace),
      path = pathlib.Path(".").resolve(),
      marker_style = [
        dict(color = 'darkslateblue', size = 10),
        dict(size = 10),
        # dict(color = 'goldenrod', size = 10, symbol = "circle"),
      ],
    )
  return

def motivational_example_fig(bert_db, fixed_bert, clgen_db):
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

  clgen_corpus = [s for s in clgen_db.correct_samples]
  # bert_datapoints = bert_db.get_by_ids([143826, 146576, 144315])
  # bert_datapoints = bert_db.get_by_ids([146576])
  bert_datapoints = [s for s in bert_db.correct_samples]
  fixed_bert = [s for s in fixed_bert.correct_samples]
  data = [s for s in bert_db.correct_samples]

  groups = {
    # "BenchPress Examples": {'data': [], 'names': []},
    "Rodinia Benchmarks": {'data': [], 'names': []},
    "CLgen samples": {'data': [], 'names': []},
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

  # unique = set()
  # for s in bert_datapoints + fixed_bert:
  #   feats = extractor.RawToDictFeats(s.feature_vector)
  #   if feature_space in feats:
  #     mon.register((feats[feature_space], "bert sample"))
  #     if "{}-{}".format(feats[feature_space]['comp'], feats[feature_space]['mem']) not in unique:
  #       groups["BenchPress Examples"]["data"].append([feats[feature_space]['comp'], feats[feature_space]['mem']])
  #       groups["BenchPress Examples"]['names'].append("")
  #       unique.add("{}-{}".format(feats[feature_space]['comp'], feats[feature_space]['mem']))
  mon.plot()
  plotter.GroupScatterPlot(
    groups = groups,
    title = "",
    plot_name = "motivational_example_compvsmem",
    path = pathlib.Path(".").resolve(),
    x_name = "# Computational Instructions",
    y_name = "# Memory Instructions",
    marker_style = [
      # dict(color = 'skyblue', size = 10, symbol = "cross"),
      # dict(color = 'firebrick', size = 10, symbol = "cross"),
      dict(color = 'darkslateblue', size = 10, symbol = "diamond-open", line = dict(width = 4)),
      dict(color = 'firebrick', size = 10, symbol = "circle"),
    ],
  )
  return

def benchpress_vs_clgen_fig(bert_db, clgen_db):

  clgen_ntoks = clgen_db.get_compilable_num_tokens
  clgen_num_insts = [x[1]["InstCountFeatures"]["TotalInsts"] for x in clgen_db.get_samples_features if "InstCountFeatures" in x[1]]

  data = bert_db.get_compilable_num_tokens
  bert_num_insts = [x[1]["InstCountFeatures"]["TotalInsts"] for x in bert_db.get_samples_features if "InstCountFeatures" in x[1]]

  plotter.RelativeDistribution(
    x = ["BenchPress", "CLgen"],
    y = [data, clgen_ntoks],
    plot_name = "token_relative_distribution",
    path = pathlib.Path(".").resolve(),
    x_name = "Token Length",
  )

  plotter.RelativeDistribution(
    x = ["BenchPress", "CLgen"],
    y = [bert_num_insts, clgen_num_insts],
    plot_name = "numinst_relative_distribution",
    path = pathlib.Path(".").resolve(),
    x_name = "LLVM IR Instructions Length (-O1)",
  )

def get_size_distribution():

  gdb = active_feed_database.ActiveFeedDatabase("sqlite:///{}".format(str("/home/fivosts/PhD/Code/clgen/results/BERT/Grewe/merged_active_feed_database.db")))
  adb = active_feed_database.ActiveFeedDatabase("sqlite:///{}".format(str("/home/fivosts/PhD/Code/clgen/results/BERT/Autophase/merged_active_feed_database.db")))
  idb = active_feed_database.ActiveFeedDatabase("sqlite:///{}".format(str("/home/fivosts/PhD/Code/clgen/results/BERT/Instcount/merged_active_feed_database.db")))

  gdata = [(c.num_tokens, c.generation_id) for c in gdb.get_data]
  adata = [(c.num_tokens, c.generation_id) for c in adb.get_data]
  idata = [(c.num_tokens, c.generation_id) for c in idb.get_data]

  m = monitors.CategoricalHistoryMonitor(pathlib.Path(".").resolve(), "size_per_gen")
  for x in gdata + adata + idata:
    m.register((x[1], x[0]))
  m.plot()
  return