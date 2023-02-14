# coding=utf-8
# Copyright 2023 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Analysis of Human or AI experiment.
"""
import pathlib
import json

from deeplearning.benchpress.experiments import public
from deeplearning.benchpress.experiments.turing import db
from deeplearning.benchpress.util import distributions
from deeplearning.benchpress.util import plotter

@public.evaluator
def HumanLikenessAnalysis(**kwargs) -> None:
  """
  Analyse Human or AI experiment.
  """
  workspace = kwargs.get("workspace_path")
  str_path = kwargs.get("human_likeness_data")
  path = pathlib.Path(str_path).resolve()
  if not path.exists():
    raise FileNotFoundError(path)
  data = db.TuringDB(url = "sqlite:///{}".format(path), must_exist = True)

  """
  Evaluate the results from the database.
  """
  """
  1. Get distribution of scores per dataset (distribute score per user / per dataset.)
    a. Get average score per dataset group.
    b. Get confidence interval.
  """
  prediction_distr = data.get_prediction_distr()
  labels = {
    "engineer": {
      "human": [[], []],
      "AI"   : [[], []]
    },
    "non-engineer": {
      "human": [[], []],
      "AI"   : [[], []]
    }
  }
  for label in labels.keys():
    for dset, values in prediction_distr[label].items():
      if values["predictions"]["human"] > 0:
        labels[label]["human"][0].append(dset)
        labels[label]["human"][1].append(values["predictions"]["human"])
      if values["predictions"]["robot"] > 0:
        labels[label]["AI"][0].append(dset)
        labels[label]["AI"][1].append(values["predictions"]["robot"])
    plotter.GrouppedBars(
      labels[label],
      plot_name = "{}_scores_per_set".format(label),
      path = workspace / "scores_per_set" / label,
    )
  # plotter.GrouppedBars(
  #   {
  #     label: [
  #       labels["engineer"][label][0] + labels["non-engineer"][label][0],
  #       labels["engineer"][label][1] + labels["non-engineer"][label][1]
  #     ]
  #     for label in labels["engineer"].keys()
  #   },
  #   plot_name = "Total_scores_per_set",
  #   path = workspace / "scores_per_set",
  # )
  unique_datasets = set(prediction_distr["engineer"].keys())
  unique_datasets.update(set(prediction_distr["non-engineer"].keys()))
  ## Distributions
  user_prediction_distr = data.get_user_prediction_distr()
  distrs = {
    "engineer": {},
    "non-engineer": {},
  }
  for label in labels.keys():
    for dset in unique_datasets:
      distrs[label][dset] = distributions.GenericDistribution(
        [
          int(100 * user[dset]["predictions"]["human"] / (user[dset]["predictions"]["robot"] + user[dset]["predictions"]["human"]))
          for user in user_prediction_distr[label] if dset in user
        ],
        log_path = workspace / "distributions" / label / dset,
        set_name = "{}_{}_distrib".format(label, dset)
      )
      distrs[label][dset].plot()
  """
  2. Conditioned probabilities:
    a. Score distribution on robots, given score on human.
    b. Score distribution on human, given score on robots.
    c. Score distribution on human and robots, given total ratio of human/robot selections.
  """

  """
  3. Measure correlation between score on human and score on GitHub.
  Plot scatter:
  x axis: Github score
  y axis: AI-dataset score.

  One datapoint: One user that has given answers to both Github and AI-dataset.
  """
  ai_datasets = set([x for x in unique_datasets if x != "GitHub"])
  correlation_data_steps = {}
  num_predictions = 2
  while True:
    correlation_data = {
      "engineer": {ai: {'data': [], 'names': [], 'frequency': []} for ai in ai_datasets},
      "non-engineer": {ai: {'data': [], 'names': [], 'frequency': []} for ai in ai_datasets}
      ,
    }
    keep_looping = False
    for label in labels.keys():
      for user in user_prediction_distr[label]:
        total = sum([x['predictions']['human'] + x['predictions']['robot'] for x in user.values()])
        for ai_set in ai_datasets:
          if ai_set in user and "GitHub" in user and total >= num_predictions:
            keep_looping = True
            dp = [
              user["GitHub"]["predictions"]["human"] / (user["GitHub"]["predictions"]["human"] + user["GitHub"]["predictions"]["robot"]),
              user[ai_set]["predictions"]["robot"] / (user[ai_set]["predictions"]["robot"] + user[ai_set]["predictions"]["human"])
            ]
            # if ai_set not in correlation_data[label]:
            #   correlation_data[label][ai_set] = {
            #     'data': [dp],
            #     'names': [""],
            #     'frequency': [1],
            #   }
            # else:
            if dp in correlation_data[label][ai_set]['data']:
              idx = correlation_data[label][ai_set]['data'].index(dp)
              correlation_data[label][ai_set]['frequency'][idx] += 1
            else:
              correlation_data[label][ai_set]['data'].append(dp)
              correlation_data[label][ai_set]['names'].append("")
              correlation_data[label][ai_set]['frequency'].append(1)
    if not keep_looping:
      break
    else:
      correlation_data_steps[num_predictions] = correlation_data
      num_predictions += 1

  for label in correlation_data.keys():
    correlation_data = {
      "x=y": {
        'data': [[x/100, x/100] for x in range(0, 105, 5)],
        'names': [[""] for x in range(0, 105, 5)],
        'frequency': [1 for x in range(0, 105, 5)]
      }
    }
    step_cov_corrs = {}
    for step in correlation_data_steps.keys():
      if len([x for y in correlation_data_steps[step][label].values() for x in y['data']]) > 0:
        correlation_data_steps[step][label].update(correlation_data)
    """
    Print the distribution of scores on AI given scores on Github.
    """
    plotter.SliderGroupScatterPlot(
      {
        s: v[label]
        for s, v in correlation_data_steps.items()
        if len([x for y in v[label].values() for x in y['data']]) > 0
      },
      "AI_vs_Human_correlation",
      path = workspace / "score_correlation" / label / "scatter",
      x_name = "Score on GitHub",
      y_name = "Score on AI",
      **kwargs,
    )
    averages = {}
    for name, values in correlation_data_steps[2][label].items():
      if name == "x=y":
        continue
      averages[name] = {}
      for dp in values["data"]:
        x, y = dp
        if x not in averages[name]:
          averages[name][x] = [y]
        else:
          averages[name][x].append(y)
      averages[name] = [[x, sum(y) / len(y)] for x, y in averages[name].items()]
      averages[name] = sorted(averages[name], key = lambda x: x[0])
    """
    Print the average distribution of scores in AI given scores on Github.
    """
    x = [[x[0] for x in data] for dname, data in averages.items()]
    y = [[y[1] for y in data] for dname, data in averages.items()]
    names = list(averages.keys())
    plotter.MultiScatterLine(
      x = x,
      y = y,
      names = names,
      plot_name = "Avg_AI_vs_Human_correlation",
      path = workspace / "score_correlation" / label / "scatter_avg",
      x_name = "Score on GitHub",
      y_name = "Avg Score on AI",
      **kwargs,
    )
    """
    Find the covariance and correlation between score on each AI and score on GitHub.
    """
    for step_id, corr_data in correlation_data_steps.items():
      step_cov_corrs[step_id] = {
        'covariance': ([], []),
        'correlation': ([], []),
      }
      for name, values in corr_data[label].items():
        if name == "x=y":
          continue
        xx = [x for x, _ in values["data"]]
        yy = [y for _, y in values["data"]]
        n = name
        if len(xx) > 0 and len(yy) > 0:
          gitd = distributions.GenericDistribution(
            [int(100*i) for i in xx],
            workspace / "score_correlation" / label / "distr",
            set_name = "score_on_git_with_{}_distr".format(n)
          )
          aid = distributions.GenericDistribution(
            [int(i*100) for i in yy],
            workspace / "score_correlation" / label / "distr",
            set_name = "score_on_{}_distr".format(n)
          )
          # gitd.plot()
          # aid.plot()
          # (aid - gitd).plot()
          step_cov_corrs[step_id]['covariance'][0].append(n)
          step_cov_corrs[step_id]['covariance'][1].append(gitd.cov(aid))
          step_cov_corrs[step_id]['correlation'][0].append(n)
          step_cov_corrs[step_id]['correlation'][1].append(gitd.corr(aid))

    plotter.SliderGrouppedBars(
      {
        k: {'covariance': v['covariance']}
        for k, v in step_cov_corrs.items()
        if len(v['covariance'][0]) > 0
      },
      plot_name = "Cov_AI_vs_Human",
      path = workspace / "score_correlation" / label / "stats",
      **kwargs,
    )
    plotter.SliderGrouppedBars(
      {
        k: {'correlation': v['correlation']}
        for k, v in step_cov_corrs.items()
        if len(v['correlation'][0]) > 0
      },
      plot_name = "Corr_AI_vs_Human",
      path = workspace / "score_correlation" / label / "stats",
      **kwargs,
    )
  return
