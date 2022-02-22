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
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.experiments import mutec
from deeplearning.clgen.experiments import srciror
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.experiments.grewe import api as grewe_api

from deeplearning.clgen.util import logging as l

from absl import app, flags

FLAGS = flags.FLAGS

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
  def data(self) -> typing.List[str]:
    """
    Get concatenated data of all databases.
    """
    if self.data:
      return self.data
    else:
      self.data = []
      for db in self.databases:
        if self.db_type == encoded.EncodedContentFiles or self.db_type == clsmith.CLSmithDatabase:
          self.data += db.get_data(self.size_limit)
        else:
          self.data += db.get_data

  def __init__(self, group_name: str, db_type: str, databases: typing.List[pathlib.Path], tokenizer = None, size_limit: int = None):
    self.group_name = group_name
    self.db_type = {
      "SamplesDatabase"    : samples_database.SamplesDatabase,
      "ActiveFeedDatabase" : active_feed_database.ActiveFeedDatabase,
      "EncodedContentFiles": encoded.EncodedContentFiles,
      "CLSmithDatabase"    : clsmith.CLSmithDatabase,
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
        db_feats = db.get_features(self.tokenizer, self.size_limit) if (self.db_type == encoded.EncodedContentFiles or self.db_type == clsmith.CLSmithDatabase) else db.get_features
        for x in db_feats:
          try:
            feats = extractor.RawToDictFeats(x)
          except Exception as e:
            l.logger().warn(x)
          if feature_space in feats and feats[feature_space]:
            self.features[feature_space].append(feats[feature_space])
    return self.features[feature_space]

  def get_data_features(self, feature_space: str, use_mp = True) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """
    Get or set feature with data list of tuples.
    """
    if not self.data_features[feature_space]:
      self.data_features[feature_space] = []
      for db in self.databases:
        db_feats = db.get_data_features(self.tokenizer, self.size_limit) if (self.db_type == encoded.EncodedContentFiles or self.db_type == clsmith.CLSmithDatabase) else db.get_data_features
        if use_mp:
          try:
            pool = multiprocessing.Pool()
            for inp, feats in tqdm.tqdm(zip(db_feats, pool.imap_unordered(workers.ContentFeat, db_feats)), total = len(db_feats), desc = "{} data".format(self.group_name)):
              if len(inp) == 2:
                src, _ = inp
                include = ""
              else:
                src, include, _ = inp
              if feature_space in feats and feats[feature_space]:
                self.data_features[feature_space].append((src, include, feats[feature_space]))
            pool.close()
          except Exception as e:
            l.logger().error(e)
            pool.terminate()
            raise e
        else:
          for inp in tqdm.tqdm(db_feats, total = len(db_feats), desc = "{} data".format(self.group_name)):
            feats = workers.ContentFeat(inp)
            if len(inp) == 2:
              src, _ = inp
              include = ""
            else:
              src, include, _ = inp
            if feature_space in feats and feats[feature_space]:
              self.data_features[feature_space].append((src, include, feats[feature_space]))
    return self.data_features[feature_space]

  def get_unique_data_features(self, feature_space: str, use_mp = True) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """
    Get or set feature with data list of tuples.
    """
    if not self.unique_data_features[feature_space]:
      self.unique_data_features[feature_space] = []
      visited = set()
      for db in self.databases:
        db_feats = db.get_data_features(self.tokenizer, self.size_limit) if (self.db_type == encoded.EncodedContentFiles or self.db_type == clsmith.CLSmithDatabase) else db.get_data_features
        if use_mp:
          try:
            pool = multiprocessing.Pool()
            for inp, (sha, feats) in tqdm.tqdm(zip(db_feats, pool.imap_unordered(workers.ContentHash, db_feats)), total = len(db_feats), desc = "{} unique data".format(self.group_name)):
              if len(inp) == 2:
                src, _ = inp
                include = ""
              else:
                src, include, _ = inp
              if sha not in visited:
                visited.add(sha)
                if feature_space in feats and feats[feature_space]:
                  self.unique_data_features[feature_space].append((src, include, feats[feature_space]))
          except Exception as e:
            l.logger().error(e)
            pool.terminate()
            raise e
          pool.close()
        else:
          for inp in db_feats:
            sha, feats = workers.ContentHash(inp)
            if len(inp) == 2:
              src, _ = inp
              include = ""
            else:
              src, include, _ = inp
            if sha not in visited:
              visited.add(sha)
              if feature_space in feats and feats[feature_space]:
                self.unique_data_features[feature_space].append((src, include, feats[feature_space]))
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

def AssertIfValid(config: evaluator_pb2.Evaluation):
  """
  Parse config file and check for validity.
  """
  pathlib.Path(config.workspace).resolve().mkdir(exist_ok = True, parents = True)
  for ev in config.evaluator:
    if ev.HasField("k_average_score"):
      ### KAverageScore
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
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
      # Specialized fields.
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
      ### MinScore
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
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
      # Specialized fields.
      pbutil.AssertFieldConstraint(
        ev.min_score,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.min_score.target),
      )
      pbutil.AssertFieldIsSet(ev.min_score, "feature_space")
    elif ev.HasField("analyze_target"):
      ### AnalyzeTarget
      # DB groups
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
      # Specialized fields.
      for target in ev.analyze_target.targets:
        assert target in feature_sampler.targets, target
    elif ev.HasField("log_file"):
      ### LogFile
      # DB groups
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
      ### CompMemGrewe

      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
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
      # Specialized fields.
      pbutil.AssertFieldConstraint(
        ev.comp_mem_grewe,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.comp_mem_grewe.target),
      )
    elif ev.HasField("topk_cldrive"):
      ### TopKCLDrive
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
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
      # Specialized fields.
      pbutil.AssertFieldConstraint(
        ev.topk_cldrive,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.topk_cldrive.target),
      )
      pbutil.AssertFieldIsSet(ev.topk_cldrive, "feature_space")
      pbutil.AssertFieldIsSet(ev.topk_cldrive, "cldrive_cache")
      if not pathlib.Path(ev.topk_cldrive.cldrive_cache).resolve().exists():
        l.logger().warn("CLDrive cache not found in {}. Will create one from scratch.".format(ev.topk_cldrive.cldrive_cache))
      pbutil.AssertFieldConstraint(
        ev.topk_cldrive,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
    elif ev.HasField("mutec_vs_benchpress"):
      ### MutecVsBenchPress
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
      if ev.mutec_vs_benchpress.HasField("db_group"):
        raise ValueError("db_group is a placeholder for mutec_vs_benchpress evaluator and should not be used.")
      for dbs in [ev.mutec_vs_benchpress.seed, ev.mutec_vs_benchpress.benchpress]:
        for db in ev.mutec_vs_benchpress.seed.database:
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
      # Specialized fields.
      pbutil.AssertFieldIsSet(ev.mutec_vs_benchpress, "mutec_cache")
      if not pathlib.Path(ev.mutec_vs_benchpress.mutec_cache).resolve().exists():
        l.logger().warn("Mutec cache not found in {}. Will create one from scratch.".format(ev.mutec_vs_benchpress.mutec_cache))

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
    elif ev.HasField("srciror_src_vs_benchpress"):
      ### MutecVsBenchPress
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
      if ev.srciror_src_vs_benchpress.HasField("db_group"):
        raise ValueError("db_group is a placeholder for srciror_src_vs_benchpress evaluator and should not be used.")
      for dbs in [ev.srciror_src_vs_benchpress.seed, ev.srciror_src_vs_benchpress.benchpress]:
        for db in ev.srciror_src_vs_benchpress.seed.database:
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
      # Specialized fields.
      pbutil.AssertFieldIsSet(ev.srciror_src_vs_benchpress, "srciror_src_cache")
      if not pathlib.Path(ev.srciror_src_vs_benchpress.srciror_src_cache).resolve().exists():
        l.logger().warn("Mutec cache not found in {}. Will create one from scratch.".format(ev.srciror_src_vs_benchpress.srciror_src_cache))

      pbutil.AssertFieldConstraint(
        ev.srciror_src_vs_benchpress,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.srciror_src_vs_benchpress.target),
      )
      pbutil.AssertFieldIsSet(ev.srciror_src_vs_benchpress, "feature_space")
      pbutil.AssertFieldConstraint(
        ev.srciror_src_vs_benchpress,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
      pbutil.AssertFieldConstraint(
        ev.srciror_src_vs_benchpress,
        "beam_width",
        lambda x: x > 0,
        "beam width factor must be positive",
      )
    elif ev.HasField("srciror_ir_vs_benchpress"):
      ### MutecVsBenchPress
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
      if ev.srciror_ir_vs_benchpress.HasField("db_group"):
        raise ValueError("db_group is a placeholder for srciror_ir_vs_benchpress evaluator and should not be used.")
      for dbs in [ev.srciror_ir_vs_benchpress.seed, ev.srciror_ir_vs_benchpress.benchpress]:
        for db in ev.srciror_ir_vs_benchpress.seed.database:
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
      # Specialized fields.
      pbutil.AssertFieldIsSet(ev.srciror_ir_vs_benchpress, "srciror_ir_cache")
      if not pathlib.Path(ev.srciror_ir_vs_benchpress.srciror_ir_cache).resolve().exists():
        l.logger().warn("Mutec cache not found in {}. Will create one from scratch.".format(ev.srciror_ir_vs_benchpress.srciror_ir_cache))

      pbutil.AssertFieldConstraint(
        ev.srciror_ir_vs_benchpress,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.srciror_ir_vs_benchpress.target),
      )
      pbutil.AssertFieldIsSet(ev.srciror_ir_vs_benchpress, "feature_space")
      pbutil.AssertFieldConstraint(
        ev.srciror_ir_vs_benchpress,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
      pbutil.AssertFieldConstraint(
        ev.srciror_ir_vs_benchpress,
        "beam_width",
        lambda x: x > 0,
        "beam width factor must be positive",
      )
    elif ev.HasField("generate_clsmith"):
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # Specialized fields.
      pbutil.AssertFieldIsSet(ev.generate_clsmith, "clsmith_db")
      if not pathlib.Path(ev.generate_clsmith.clsmith_db).resolve().exists():
        l.logger().warn("CLSmith samples DB not found in {}. Will create one from scratch.".format(ev.generate_clsmith.clsmith_db))
    elif ev.HasField("grewe_top_k_csv"):
      ### TopKCLDrive
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
      for dbs in ev.grewe_top_k_csv.db_group:
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
      # Specialized fields.
      pbutil.AssertFieldIsSet(ev.grewe_top_k_csv, "cldrive_cache")
      if not pathlib.Path(ev.grewe_top_k_csv.cldrive_cache).resolve().exists():
        l.logger().warn("CLDrive cache not found in {}. Will create one from scratch.".format(ev.grewe_top_k_csv.cldrive_cache))
      pbutil.AssertFieldConstraint(
        ev.grewe_top_k_csv,
        "target",
        lambda x: x in feature_sampler.targets,
        "target {} not found".format(ev.grewe_top_k_csv.target),
      )
      pbutil.AssertFieldConstraint(
        ev.grewe_top_k_csv,
        "top_k",
        lambda x: x > 0,
        "top-K factor must be positive",
      )
    elif ev.HasField("grewe_csv"):
      ### TopKCLDrive
      # Generic Fields
      pbutil.AssertFieldIsSet(config, "workspace")
      pbutil.AssertFieldIsSet(config, "tokenizer")
      if not pathlib.Path(config.tokenizer).resolve().exists():
        raise FileNotFoundError(pathlib.Path(config.tokenizer).resolve())
      # DB groups
      for dbs in ev.grewe_csv.db_group:
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
      pbutil.AssertFieldIsSet(ev.grewe_csv, "cldrive_cache")
      if not pathlib.Path(ev.grewe_csv.cldrive_cache).resolve().exists():
        l.logger().warn("CLDrive cache not found in {}. Will create one from scratch.".format(ev.grewe_csv.cldrive_cache))
    elif ev.HasField("train_grewe"):
      ### TrainGrewe
      # Generic fields
      pbutil.AssertFieldIsSet(config, "workspace")
      # CSV groups
      pbutil.AssertFieldIsSet(config, "grewe_baseline")
      p = pathlib.Path(config.grewe_baseline)
      if not p.exists():
        raise FileNotFoundError(p)
      for c in ev.train_grewe.csv:
        pbutil.AssertFieldIsSet(c.name)
        pbutil.AssertFieldIsSet(c.path)
        p = pathlib.Path(c.path)
        if not p.exists():
          raise FileNotFoundError(p)
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

def main(config: evaluator_pb2.Evaluation):
  """
  Run the evaluators iteratively.
  """
  evaluation_map = {
    evaluator_pb2.LogFile                 : log_file.LogFile,
    evaluator_pb2.KAverageScore           : distance_score.KAverageScore,
    evaluator_pb2.MinScore                : distance_score.MinScore,
    evaluator_pb2.AnalyzeTarget           : benchmark_analysis.AnalyzeTarget,
    evaluator_pb2.CompMemGrewe            : comp_vs_mem.CompMemGrewe,
    evaluator_pb2.TopKCLDrive             : cldrive.TopKCLDrive,
    evaluator_pb2.MutecVsBenchPress       : mutec.MutecVsBenchPress,
    evaluator_pb2.SRCIROR_srcVsBenchPress : srciror.SRCIRORVsBenchPress,
    evaluator_pb2.SRCIROR_IRVsBenchPress  : srciror.SRCIRORVsBenchPress,
    evaluator_pb2.GenerateCLSmith         : clsmith.GenerateCLSmith,
    evaluator_pb2.GreweTopKCSV            : grewe_api.GreweTopKCSV,
    evaluator_pb2.GreweCSV                : grewe_api.GreweCSV,
  }
  db_cache       = {}
  target_cache   = {}
  feature_spaces = []
  for ev in config.evaluator:
    kw_args = {
      "db_groups"      : [],
      "tokenizer"      : tokenizers.TokenizerBase.FromFile(pathlib.Path(config.tokenizer).resolve()) if config.HasField("tokenizer") else None,
      "workspace_path" : pathlib.Path(config.workspace).resolve() if config.HasField("workspace") else None,
    }
    if ev.HasField("k_average_score"):
      sev = ev.k_average_score
      kw_args['top_k'] = sev.top_k
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("min_score"):
      sev = ev.min_score
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("analyze_target"):
      sev = ev.analyze_target
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("log_file"):
      sev = ev.log_file
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("comp_mem_grewe"):
      sev = ev.comp_mem_grewe
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("topk_cldrive"):
      sev = ev.topk_cldrive
      kw_args['top_k']         = sev.top_k
      kw_args['cldrive_cache'] = sev.cldrive_cache
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("mutec_vs_benchpress"):
      sev = ev.mutec_vs_benchpress
      kw_args['top_k']       = sev.top_k
      kw_args['mutec_cache'] = sev.mutec_cache
      kw_args['beam_width']  = sev.beam_width
      # Gather target benchmarks and cache them
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
      for name, dbs in [('seed', sev.seed), ('benchpress', sev.benchpress)]:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args[name] = db_cache[key]
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("srciror_src_vs_benchpress") or ev.HasField("srciror_ir_vs_benchpress"):
      if ev.HasField("srciror_src_vs_benchpress"):
        sev = ev.srciror_src_vs_benchpress
        kw_args['srciror_cache']  = sev.srciror_src_cache
        kw_args['mutation_level'] = "src"
      else:
        sev = ev.srciror_ir_vs_benchpress
        kw_args['srciror_cache'] = sev.srciror_ir_cache
        kw_args['mutation_level'] = "IR"
  
      kw_args['top_k']         = sev.top_k
      kw_args['srciror_cache'] = sev.srciror_src_cache
      kw_args['beam_width']    = sev.beam_width
      # Gather target benchmarks and cache them
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
      for name, dbs in [('seed', sev.seed), ('benchpress', sev.benchpress)]:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args[name] = db_cache[key]
      # Gather feature spaces if applicable.
      if sev.HasField("feature_space"):
        kw_args['feature_space'] = sev.feature_space
      # Gather plotter configuration
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config

    elif ev.HasField("generate_clsmith"):
      sev = ev.generate_clsmith
      kw_args['clsmith_path'] = sev.clsmith_db

    elif ev.HasField("grewe_top_k_csv"):
      sev = ev.grewe_top_k_csv
      kw_args['top_k'] = sev.top_k
      kw_args['cldrive_cache'] = sev.cldrive_cache
      # Gather target benchmarks and cache them
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
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      kw_args['feature_space'] = "GreweFeatures"

    elif ev.HasField("grewe_csv"):
      sev = ev.grewe_csv
      kw_args['cldrive_cache'] = sev.cldrive_cache
      # Gather target benchmarks and cache them
      for dbs in sev.db_group:
        key = dbs.group_name + ''.join(dbs.database)
        if key not in db_cache:
          size_limit = dbs.size_limit if dbs.HasField("size_limit") else None
          db_cache[key] = DBGroup(dbs.group_name, dbs.db_type, dbs.database, tokenizer = kw_args['tokenizer'], size_limit = size_limit)
        kw_args['db_groups'].append(db_cache[key])
      kw_args['feature_space'] = "GreweFeatures"

    elif ev.HasField("train_grewe"):
      sev = ev.train_grewe
      if sev.HasField("plot_config"):
        kw_args['plot_config'] = sev.plot_config
      kw_args['grewe_baseline'] = pathlib.Path(sev.grewe_baseline).resolve()
      kw_args['csv_groups'] = []
      for c in sev.csv:
        kw_args['csv_groups'].append({'name': c.name, 'path': pathlib.Path(c.path).resolve()})

    else:
      raise NotImplementedError(ev)

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
        bc  = workers.SortedSrcDistances(bert_corpus,        benchmark.features, self.feature_space)
        cc  = workers.SortedSrcDistances(clgen_corpus,       benchmark.features, self.feature_space)
        gc  = workers.SortedSrcDistances(git_corpus,         benchmark.features, self.feature_space)
        rgc = workers.SortedSrcDistances(reduced_git_corpus, benchmark.features, self.feature_space)
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
      rgc = workers.SortedSrcDistances(reduced_git, benchmark.features[self.feature_space], self.feature_space)
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
