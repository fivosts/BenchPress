"""
Target benchmark analysis evaluator.
"""
import tqdm

from deeplearning.clgen.experiments import public
from deeplearning.clgen.experiments import clsmith
from deeplearning.clgen.experiments import workers
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.util import plotter

@public.evaluator
def AnalyzeTarget(**kwargs) -> None:
  """
  Analyze requested target benchmark suites.
  """
  targets   = kwargs.get('targets')
  tokenizer = kwargs.get('tokenizer')
  workspace_path = kwargs.get('workspace_path')
  raise NotImplementedError
  return

@public.evaluator
def FeaturesDistribution(**kwargs) -> None:
  """
  Plot distribution of features per feature dimension per database group.
  """
  db_groups      = kwargs.get('db_groups')
  target         = kwargs.get('targets')
  feature_space  = kwargs.get('feature_space')
  top_k          = kwargs.get('top_k')
  unique_code    = kwargs.get('unique_code', False)
  plot_config    = kwargs.get('plot_config')
  workspace_path = kwargs.get('workspace_path')
  data = {}

  # You need this if you want to have the same (github) baseline but when github is not plotted.
  reduced_git = None
  for dbg in db_groups:
    if dbg.group_name == "GitHub-768-inactive" or dbg.group_name == "GitHub-768":
      reduced_git = dbg.get_data_features(feature_space)
      break

  benchmarks = target.get_benchmarks(feature_space, reduced_git_corpus = reduced_git)
  for idx, dbg in enumerate(db_groups):
    if dbg.group_name == "GitHub-768-inactive":
      # Skip baseline DB group.
      continue
    if not (dbg.db_type == samples_database.SamplesDatabase or dbg.db_type == encoded.EncodedContentFiles or dbg.db_type == clsmith.CLSmithDatabase):
      raise ValueError("Scores require SamplesDatabase or EncodedContentFiles but received", dbg.db_type)
    data[dbg.group_name] = {}
    for benchmark in tqdm.tqdm(benchmarks, total = len(benchmarks), desc = "Benchmarks"):
      # Find shortest distances.
      if unique_code:
        get_data = lambda x: dbg.get_unique_data_features(x)
      else:
        get_data = lambda x: dbg.get_data_features(x)

      if idx == 0:
        if targe.targett not in data:
          data[target.target] = {}
        for k, v in benchmark.features.items():
          if k not in data[target.target]:
            data[target.target][k] = [v]
          else:
            data[target.target][k].append(v)

      ret = workers.SortedSrcFeatsDistances(get_data(feature_space), benchmark.features, feature_space)[:top_k]
      for _, _, fvec, _ in ret:
        for k, v in fvec.items():
          if k not in data[dbg.group_name]:
            data[dbg.group_name][k] = [v]
          else:
            data[dbg.group_name][k].append(v)

  plotter.GrouppedViolins(
    data = data,
    plot_name = "feat_distr_{}_dist_{}_{}".format(top_k, feature_space.replace("Features", " Features"), '-'.join([dbg.group_name for dbg in db_groups])),
    path = workspace_path,
    **plot_config if plot_config else {},
  )
  return
