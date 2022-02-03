"""
Evaluators - result fetchers for samples across different techniques.
"""
import typing
import io
import glob
import progressbar
import json
import os
import tempfile
import subprocess
import pickle
import pathlib
import tqdm
import multiprocessing
import functools
import sklearn
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from deeplearning.clgen.proto import evaluator_pb2
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import active_feed_database
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.util import monitors
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.experiments import log_file
from deeplearning.clgen.experiments import benchmark_analysis
from deeplearning.clgen.experiments import distance_score
from deeplearning.clgen.experiments import comp_vs_mem
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.experiments import mutec

from deeplearning.clgen.util import logging as l

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "evaluator_config",
  "",
  "Set path to evaluator config file",
)

def AssertIfValid(config: evaluator_pb2.Evaluation):
  """
  Parse config file and check for validity.
  """
  pbutil.AssertFieldIsSet(config, "workspace")
  pbutil.AssertFieldIsSet(config, "tokenizer")
  if not pathlib.Path(config.tokenizer).resolve().exists():
    raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
  pathlib.Path(config.workspace).resolve().mkdir(exist_ok = True, parents = True)
  for ev in config.evaluator:
    if ev.HasField("k_average_score"):
      for dbs in ev.k_average_score.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
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
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
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
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
      for target in ev.analyze_target.targets:
        assert target in feature_sampler.targets, target
    elif ev.HasField("log_file"):
      for dbs in ev.log_file.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
    elif ev.HasField("comp_mem_grewe"):
      for dbs in ev.comp_mem_grewe.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
      pbutil.AssertFieldConstraint(
        ev.comp_mem_grewe,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.comp_mem_grewe.target),
      )
    elif ev.HasField("topk_cldrive"):
      for dbs in ev.topk_cldrive.db_group:
        for db in dbs.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )
      pbutil.AssertFieldConstraint(
        ev.topk_cldrive,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.topk_cldrive.target),
      )
      pbutil.AssertFieldIsSet(ev.topk_cldrive, "feature_space")
      pbutil.AssertFieldIsSet(ev.topk_cldrive, "cldrive_cache")
      if not pathlib.Path(ev.topk_cldrive.cldrive_cache).resolve().exists():
        l.logger().warn("CLDrive cache not found in {}. Will create one from scratch.".format(ev.topk_cldrive))
      pbutil.AssertFieldConstraint(
        ev.topk_cldrive,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
    elif ev.HasField("mutec_vs_benchpress"):
      if ev.mutec_vs_benchpress.HasField("db_group"):
        raise ValueError("db_group is a placeholder for mutec_vs_benchpress evaluator and should not be used.")
      for dbs in [ev.mutec_vs_benchpress.github, ev.mutec_vs_benchpress.benchpress]:
        for db in ev.mutec_vs_benchpress.github.database:
          p = pathlib.Path(db).resolve()
          if not p.exists():
            raise FileNotFoundError(p)
        if dbs.HasField("size_limit"):
          pbutil.AssertFieldConstraint(
            dbs,
            "size_limit",
            lambda x : x > 0,
            "Size limit must be a positive integer, {}".format(dbs.size_limit)
          )

      pbutil.AssertFieldIsSet(ev.mutec_vs_benchpress, "mutec_cache")
      if not pathlib.Path(ev.mutec_vs_benchpress.mutec_cache).resolve().exists():
        l.logger().warn("Mutec cache not found in {}. Will create one from scratch.".format(ev.mutec_vs_benchpress))

      pbutil.AssertFieldConstraint(
        ev.mutec_vs_benchpress,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.mutec_vs_benchpress.target),
      )
      pbutil.AssertFieldIsSet(ev.mutec_vs_benchpress, "feature_space")
      pbutil.AssertFieldConstraint(
        ev.mutec_vs_benchpress,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
      pbutil.AssertFieldConstraint(
        ev.mutec_vs_benchpress,
        "beam_width",
        lambda x: x > 0,
        "beam width factor must be positive",
      )
    else:
      raise ValueError(ev)
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

def ContentHash_worker(db_feat: typing.Tuple[str, str]) -> typing.Tuple[str, typing.Dict[str, float]]:
  """
  Multiprocessing Worker calculates contentfile hash
  of file and returns it.
  """
  src, feats = db_feat
  try:
    return opencl.ContentHash(src), extractor.RawToDictFeats(feats)
  except Exception as e:
    l.logger().warn(e)
    return None

def ContentFeat_worker(db_feat: typing.Tuple[str, str]) -> typing.Dict[str, float]:
  """
  Multiprocessing Worker calculates contentfile hash
  of file and returns it.
  """
  _, feats = db_feat
  try:
    return extractor.RawToDictFeats(feats)
  except Exception as e:
    l.logger().warn(e)
    return None

def SortedDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                    target_features: typing.Dict[str, float],
                    feature_space: str
                    ) -> typing.List[float]:
  """
  Return list of pairs of source with respective euclidean distances from target features in ascending order.
  """
  return sorted([feature_sampler.calculate_distance(dp, target_features, feature_space) for _, dp in data])

def SortedSrcDistances(data: typing.List[typing.Tuple[str, typing.Dict[str, float]]],
                       target_features: typing.Dict[str, float],
                       feature_space: str
                       ) -> typing.List[typing.Tuple[str, float]]:
  """
  Return list of euclidean distances from target features in ascending order.
  """
  return sorted([(src, feature_sampler.calculate_distance(dp, target_features, feature_space)) for src, dp in data], key = lambda x: x[1])

class DBGroup(object):
  """
  Class representation of a group of databases evaluated.
  """
  @property
  def data(self) -> typing.List[str]:
    """
    Get concatenated data of all databases.
    """
    if self.data:
      return self.data
    else:
      self.data = []
      for db in self.databases:
        if self.db_type == encoded.EncodedContentFiles:
          self.data += db.get_data(self.size_limit)
        else:
          self.data += db.get_data

  def __init__(self, group_name: str, db_type: str, databases: typing.List[pathlib.Path], tokenizer = None, size_limit: int = None):
    self.group_name = group_name
    self.db_type = {
      "SamplesDatabase"    : samples_database.SamplesDatabase,
      "ActiveFeedDatabase" : active_feed_database.ActiveFeedDatabase,
      "EncodedContentFiles": encoded.EncodedContentFiles,
    }[db_type]
    self.databases            = [self.db_type("sqlite:///{}".format(pathlib.Path(p).resolve()), must_exist = True) for p in databases]
    self.features             = {ext: None for ext in extractor.extractors.keys()}
    self.data_features        = {ext: None for ext in extractor.extractors.keys()}
    self.unique_data_features = {ext: None for ext in extractor.extractors.keys()}
    self.tokenizer            = tokenizer
    self.size_limit           = size_limit
    return

  def get_features(self, feature_space: str) -> typing.List[typing.Dict[str, float]]:
    """
    Get or set and get features for a specific feature space.
    """
    if not self.features[feature_space]:
      self.features[feature_space] = []
      for db in self.databases:
        db_feats = db.get_features(self.tokenizer, self.size_limit) if self.db_type == encoded.EncodedContentFiles else db.get_features
        for x in db_feats:
          try:
            feats = extractor.RawToDictFeats(x)
          except Exception as e:
            l.logger().warn(x)
          if feature_space in feats and feats[feature_space]:
            self.features[feature_space].append(feats[feature_space])
    return self.features[feature_space]

  def get_data_features(self, feature_space: str) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """
    Get or set feature with data list of tuples.
    """
    if not self.data_features[feature_space]:
      self.data_features[feature_space] = []
      for db in self.databases:
        db_feats = db.get_data_features(self.tokenizer, self.size_limit) if self.db_type == encoded.EncodedContentFiles else db.get_data_features
        pool = multiprocessing.Pool()
        try:
          for (src, _), feats in tqdm.tqdm(zip(db_feats, pool.imap_unordered(ContentFeat_worker, db_feats)), total = len(db_feats), desc = "{} data".format(self.group_name)):
            if feature_space in feats and feats[feature_space]:
              self.data_features[feature_space].append((src, feats[feature_space]))
        except Exception as e:
          l.logger().error(e)
          pool.terminate()
          raise e
        pool.close()
    return self.data_features[feature_space]

  def get_unique_data_features(self, feature_space: str) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """
    Get or set feature with data list of tuples.
    """
    if not self.unique_data_features[feature_space]:
      self.unique_data_features[feature_space] = []
      visited = set()
      for db in self.databases:
        db_feats = db.get_data_features(self.tokenizer, self.size_limit) if self.db_type == encoded.EncodedContentFiles else db.get_data_features
        pool = multiprocessing.Pool()
        try:
          for (src, _), (sha, feats) in tqdm.tqdm(zip(db_feats, pool.imap_unordered(ContentHash_worker, db_feats)), total = len(db_feats), desc = "{} unique data".format(self.group_name)):
            if sha not in visited:
              visited.add(sha)
              if feature_space in feats and feats[feature_space]:
                self.unique_data_features[feature_space].append((src, feats[feature_space]))
        except Exception as e:
          l.logger().error(e)
          pool.terminate()
          raise e
        pool.close()
    return self.unique_data_features[feature_space]

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
    self.benchmarks    = {ext: [] for ext in extractor.extractors.keys()}
    l.logger().info("Loaded {} {} benchmarks".format(len(self.benchmark_cfs), self.target))
    return

  def get_benchmarks(self, feature_space: str):
    """
    Get or set and get benchmarks with their features for a feature space.
    """
    if not self.benchmarks[feature_space]:
      for p, k, h in self.benchmark_cfs:
        features = extractor.ExtractFeatures(k, [feature_space], header_file = h, use_aux_headers = False)
        if features[feature_space]:
          self.benchmarks[feature_space].append(
            Benchmark(
                p,
                p.name,
                k,
                features[feature_space],
              )
          )
      self.benchmarks[feature_space] = feature_sampler.resolve_benchmark_names(self.benchmarks[feature_space])
      l.logger().info("Extracted features for {} {} benchmarks".format(len(self.benchmarks[feature_space]), self.target))
    return self.benchmarks[feature_space]

def main(config: evaluator_pb2.Evaluation):
  """
  Run the evaluators iteratively.
  """
  evaluation_map = {
    evaluator_pb2.LogFile           : log_file.LogFile,
    evaluator_pb2.KAverageScore     : distance_score.KAverageScore,
    evaluator_pb2.MinScore          : distance_score.MinScore,
    evaluator_pb2.AnalyzeTarget     : benchmark_analysis.AnalyzeTarget,
    evaluator_pb2.CompMemGrewe      : comp_vs_mem.CompMemGrewe,
    evaluator_pb2.TopKCLDrive       : cldrive.TopKCLDrive,
    evaluator_pb2.MutecVsBenchPress : mutec.MutecVsBenchPress
  }
  db_cache       = {}
  target_cache   = {}
  feature_spaces = []
  for ev in config.evaluator:
    kw_args = {
      "db_groups"      : [],
      "tokenizer"      : tokenizers.TokenizerBase.FromFile(pathlib.Path(config.tokenizer).resolve()),
      "workspace_path" : pathlib.Path(config.workspace).resolve(),
    }
    if ev.HasField("k_average_score"):
      sev = ev.k_average_score
      kw_args['top_k'] = sev.top_k
    elif ev.HasField("min_score"):
      sev = ev.min_score
    elif ev.HasField("analyze_target"):
      sev = ev.analyze_target
    elif ev.HasField("log_file"):
      sev = ev.log_file
    elif ev.HasField("comp_mem_grewe"):
      sev = ev.comp_mem_grewe
    elif ev.HasField("topk_cldrive"):
      sev = ev.topk_cldrive
      kw_args['top_k']         = sev.top_k
      kw_args['cldrive_cache'] = sev.top_k
    elif ev.HasField("mutec_vs_benchpress"):
      sev = ev.mutec_vs_benchpress
      kw_args['top_k']      = sev.top_k
      kw_args['mutec_cache']      = sev.mutec_cache
      kw_args['beam_width'] = sev.beam_width
      for name, dbs in [('github', sev.github), ('benchpress', sev.benchpress)]:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args[name] = db_cache[key]
    else:
      raise NotImplementedError(ev)

    # Gather database groups and cache them.
    if not ev.HasField("mutec_vs_benchpress"):
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
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
    # Gather plotter configuration
    if sev.HasField("plot_config"):
      kw_args['plot_config'] = sev.plot_config

    evaluation_map[type(sev)](**kw_args)
  return

def initMain(*args, **kwargs):
  l.initLogger(name = "evaluators")
  config = ConfigFromFlags()
  main(config)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)

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
        bc  = SortedSrcDistances(bert_corpus,        benchmark.features, self.feature_space)
        cc  = SortedSrcDistances(clgen_corpus,       benchmark.features, self.feature_space)
        gc  = SortedSrcDistances(git_corpus,         benchmark.features, self.feature_space)
        rgc = SortedSrcDistances(reduced_git_corpus, benchmark.features, self.feature_space)
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

  def benchmark_stats(self, reduced_git):
    """
    [Deprecated] Prints stats of source codes for reduced git corpus.
    """
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
      rgc = SortedSrcDistances(reduced_git, benchmark.features[self.feature_space], self.feature_space)
      if rgc[1] > 0:
        final_benchmarks.append(benchmark)

    for benchmark in final_benchmarks:
      if benchmark.name not in names:
        names[benchmark.name] = [0, 0]
      else:
        names[benchmark.name][1] += 1

    l.logger().info("Analyzing {} benchmarks".format(len(final_benchmarks)))

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

def benchpress_vs_clgen_pca(bert_db, clgen_db):
  """
  PCA feature reduction for all three feature spaces for benchpress vs CLgen.
  Used in PCA plots for PLDI paper.
  """
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

def benchpress_vs_clgen_fig(bert_db, clgen_db):
  """
  PLDI paper relative distribution plots of token length and number of LLVM-IR instructions length.
  """
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
  """
  Calculates distribution of code sizes per database group.
  """
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
