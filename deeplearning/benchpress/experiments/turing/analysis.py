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
      plot_name = "Engineers_scores_per_set",
      path = workspace,
    )
  plotter.GrouppedBars(
    {
      label: [labels["engineer"][label][0], labels["engineer"][label][1]]
      for label in labels["engineer"].keys()
    },
    plot_name = "Total_scores_per_set",
    path = workspace,
  )
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
        log_path = workspace,
        set_name = "{}_{}_distrib".format(label, dset)
      )
      distrs[label][dset].plot()

  """
  2. Conditioned probabilities:
    a. Score distribution on robots, given score on human.
    b. Score distribution on human, given score on robots.
    c. Score distribution on human and robots, given total ratio of human/robot selections.
  """
  return
