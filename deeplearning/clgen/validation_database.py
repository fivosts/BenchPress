"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import sqlite3

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen import crypto
from labm8.py import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class BERTValFile(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen validation trace.

  """
  __tablename__    = "validation_traces"
  id                            : int = sql.Column(sql.Integer,    primary_key = True)
  sha256                        : str = sql.Column(sql.String(64), nullable = False, index = True)
  train_step                    : int = sql.Column(sql.Integer,    nullable = False)
  original_input                : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_original_input        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  input_ids                     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  input_mask                    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_input_ids             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_positions           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_ids                 : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_mask_lm_ids           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_weights             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_lengths             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  next_sentence_labels          : int = sql.Column(sql.Integer,    nullable = False)
  masked_lm_predictions         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_masked_lm_predictions : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  next_sentence_predictions     : int = sql.Column(sql.Integer,    nullable = False)
  num_targets                   : int = sql.Column(sql.Integer,    nullable = False)
  date_added                    : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls, 
               atomizer,
               id: int,
               train_step: int,
               original_input,
               input_ids,
               input_mask,
               masked_lm_ids,
               masked_lm_positions,
               masked_lm_weights,
               masked_lm_lengths,
               next_sentence_labels,
               masked_lm_predictions,
               next_sentence_predictions,
               ) -> typing.Dict[str, typing.Any]:

    str_original_input              = atomizer.DeatomizeIndices(original_input, ignore_token = atomizer.padToken)
    str_input_ids                   = atomizer.DeatomizeIndices(input_ids, ignore_token = atomizer.padToken)
    str_masked_lm_ids               = '\n'.join([atomizer.decoder[x] if x != atomizer.vocab['\n'] else '\\n' for x in masked_lm_ids])
    str_masked_lm_predictions       = '\n'.join([atomizer.decoder[x] if x != atomizer.vocab['\n'] else '\\n' for x in masked_lm_predictions])

    return {
      "id"                            : id,
      "sha256"                        : crypto.sha256_str(
                                              str(int(train_step)) + 
                                              str_original_input + 
                                              str_input_ids + 
                                              str_masked_lm_ids + 
                                              str_masked_lm_predictions
                                            ),
      "train_step"                    : int(train_step),
      "original_input"                : str_original_input,
      "encoded_original_input"        : ','.join([str(x) for x in original_input]),
      "input_ids"                     : str_input_ids,
      "encoded_input_ids"             : ','.join([str(x) for x in input_ids]),
      "input_mask"                    : ','.join([str(x) for x in input_mask]),
      "masked_lm_positions"           : ','.join([str(x) for x in masked_lm_positions]),
      "masked_lm_ids"                 : str_masked_lm_ids,
      "encoded_mask_lm_ids"           : ','.join([str(x) for x in masked_lm_ids]),
      "masked_lm_weights"             : ','.join([str(int(x)) for x in masked_lm_weights]),
      "masked_lm_lengths"             : ','.join([str(int(x)) for x in masked_lm_lengths]),
      "next_sentence_labels"          : int(next_sentence_labels),
      "masked_lm_predictions"         : str_masked_lm_predictions,
      "encoded_masked_lm_predictions" : ','.join([str(x) for x in masked_lm_predictions]),
      "next_sentence_predictions"     : int(next_sentence_predictions),
      "num_targets"                   : list(masked_lm_ids).index(atomizer.padToken) if atomizer.padToken in list(masked_lm_ids) else len(list(masked_lm_ids)),
      "date_added"                    : datetime.datetime.utcnow(),
    }

class ValidationDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(ValidationDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    with self.Session() as s:
      count = s.query(BERTValFile).count()
    return count