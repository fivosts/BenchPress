"""A module for databases of CLgen samples."""
import datetime
import typing

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class ValResults(Base):
  __tablename__ = "validation_results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class BERTValFile(Base, sqlutil.ProtoBackedMixin):
  """
    A database entry representing a CLgen validation trace.
  """
  __tablename__    = "validation_traces"
  id                            : int = sql.Column(sql.Integer,    primary_key = True)
  sha256                        : str = sql.Column(sql.String(64), nullable = False, index = True)
  train_step                    : int = sql.Column(sql.Integer,    nullable = False)
  original_input                : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  input_ids                     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_ids                 : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_positions           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_lengths             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_predictions         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  num_targets                   : int = sql.Column(sql.Integer,    nullable = False)
  encoded_original_input        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_input_ids             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  input_mask                    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_mask_lm_ids           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_weights             : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  encoded_masked_lm_predictions : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  next_sentence_labels          : int = sql.Column(sql.Integer,    nullable = False)
  next_sentence_predictions     : int = sql.Column(sql.Integer,    nullable = False)
  seen_in_training              : int = sql.Column(sql.Integer,    nullable = False)
  date_added                    : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls, 
               tokenizer,
               id: int,
               train_step: int,
               seen_in_training,
               original_input:            typing.List[int],
               input_ids:                 typing.List[int],
               input_mask:                typing.List[int],
               masked_lm_ids:             typing.List[int],
               masked_lm_positions:       typing.List[int],
               masked_lm_weights:         typing.List[float],
               masked_lm_lengths:         typing.List[int],
               next_sentence_labels:      typing.List[int],
               masked_lm_predictions:     typing.List[int],
               next_sentence_predictions: typing.List[int],
               ) -> typing.Dict[str, typing.Any]:

    str_original_input              = tokenizer.tokensToString(original_input, ignore_token = tokenizer.padToken, with_formatting = True)
    str_input_ids                   = tokenizer.tokensToString(input_ids,      ignore_token = tokenizer.padToken, with_formatting = True)
    str_masked_lm_ids               = '\n'.join([tokenizer.decoder[x] if ('\n' not in tokenizer.vocab or  ('\n' in tokenizer.vocab and x != tokenizer.vocab['\n'])) else '\\n' for x in masked_lm_ids])
    str_masked_lm_predictions       = '\n'.join([tokenizer.decoder[x] if ('\n' not in tokenizer.vocab or  ('\n' in tokenizer.vocab and x != tokenizer.vocab['\n'])) else '\\n' for x in masked_lm_predictions])

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
      "masked_lm_lengths"             : ','.join([str(int(x)) for x in masked_lm_lengths if x >= 0]),
      "next_sentence_labels"          : int(next_sentence_labels),
      "masked_lm_predictions"         : str_masked_lm_predictions,
      "encoded_masked_lm_predictions" : ','.join([str(x) for x in masked_lm_predictions]),
      "next_sentence_predictions"     : int(next_sentence_predictions),
      "num_targets"                   : list(masked_lm_ids).index(tokenizer.padToken) if tokenizer.padToken in list(masked_lm_ids) else len(list(masked_lm_ids)),
      "seen_in_training"              : int(seen_in_training),
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