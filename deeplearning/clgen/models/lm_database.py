"""Observation database for lm datasets."""
import typing

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from labm8.py import sqlutil

Base = declarative.declarative_base()

class LMStats(Base):
  __tablename__ = "validation_results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class LMInstance(Base, sqlutil.ProtoBackedMixin):
  """
    A database entry representing a CLgen validation trace.
  """
  __tablename__ = "masked_lm_instances"
  id                : int = sql.Column(sql.Integer,    primary_key = True, index = True)
  original_input    : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  input_ids         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  masked_lm_lengths : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

  @classmethod
  def FromArgs(cls, 
               id                :int,
               original_input    :str,
               input_ids         :str,
               masked_lm_lengths :typing.List[int],
               ) -> typing.Dict[str, typing.Any]:
    return {
      "id"                : id,
      "original_input"    : original_input,
      "input_ids"         : input_ids,
      "masked_lm_lengths" : ','.join([str(int(x)) for x in masked_lm_lengths]),
    }

class LMDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False):
    super(LMDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    with self.Session() as s:
      count = s.query(LMInstance).count()
    return count