"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import pathlib
import typing
import tqdm
import multiprocessing
import pickle

from absl import app

from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.util import distributions
from deeplearning.benchpress.experiments import workers

ENCODED_DB_PATH = "/home/foivos/unique_encoded.db"
TOKENIZER_PATH = "/home/foivos/backup_tokenizer.pkl"

def get_data_features(feature_space: str, db, tokenizer, size_limit = None) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
  """
  Get or set feature with data list of tuples.
  """
  data_features = {}
  data_features[feature_space] = []
  db_feats = db.get_data_features(tokenizer, size_limit)
  for inp in tqdm.tqdm(db_feats, total = len(db_feats), desc = "Fetch data"):
    feats = workers.ContentFeat(inp)
    if len(inp) == 2:
      src, _ = inp
      include = ""
    else:
      src, include, _ = inp
    if feature_space in feats and feats[feature_space]:
      data_features[feature_space].append((src, include, feats[feature_space]))
  return data_features[feature_space]

def main(*args):

  db = encoded.EncodedContentFiles(url = "sqlite:///{}".format(ENCODED_DB_PATH), must_exist = True)
  tokenizer = tokenizers.TokenizerBase.FromFile(pathlib.Path(TOKENIZER_PATH).resolve())

  distr = {
    "GreweFeatures": None,
    "AutophaseFeatures": None,
    "InstCountFeatures": None,
  }

  distr_768 = {
    "GreweFeatures": None,
    "AutophaseFeatures": None,
    "InstCountFeatures": None,
  }

  for fspace in {"GreweFeatures", "AutophaseFeatures", "InstCountFeatures"}:
    feat_vecs = [v for s, i, v in get_data_features(fspace, db, tokenizer)]
    flat_vals = []
    for vec in feat_vecs:
      for v in vec.values():
        try:
          flat_vals.append(4 * int(v // 4))
        except Exception:
          pass
    distr[fspace] = distributions.GenericDistribution(flat_vals, "feature_vals", fspace)
    distr[fspace].plot()

  for fspace in {"GreweFeatures", "AutophaseFeatures", "InstCountFeatures"}:
    feat_vecs = [v for s, i, v in get_data_features(fspace, db, tokenizer, 768)]
    flat_vals = []
    for vec in feat_vecs:
      for v in vec.values():
        try:
          flat_vals.append(4 * int(v // 4))
        except Exception:
          pass
    distr_768[fspace] = distributions.GenericDistribution(flat_vals, "feature_vals", "{}_768".format(fspace))
    distr_768[fspace].plot()
  return

if __name__ == "__main__":
  app.run(main)
