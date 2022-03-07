"""
Module handling Structs Database.

This database captures and handles the struct/class/union/enum
dependencies found during raw corpus preprocessing.
"""
import datetime
import sqlite3
import pathlib
import typing
import clang.cindex

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import logging as l

Base = declarative.declarative_base()

class Data(Base):
  __tablename__ = "data"
  """
  DB Table holding struct meta-data and stats.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class Struct(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representing a parsed struct.
  """
  __tablename__    = "structs"
  # entry id.
  id         : int = sql.Column(sql.Integer, primary_key = True)
  # unique, indexable content has of struct.
  sha256     : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Struct contents.
  contents   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Struct name.
  name       : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Struct fields.
  fields     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Number of fields a struct has.
  num_fields : int = sql.Column(sql.Integer, nullable = False)
  # Repo name where struct was found.
  repo_name  : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Repo ref.
  ref        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id         : int,
               contents   : str,
               num_fields : int,
               repo_name  : str,
               ref        : str,
               ) -> 'Struct':
    return Struct

class DatatypeDirectory(sqlutil.Database):
  """A database directory of C/OpenCL composite types and functions."""
  def __init__(self, url: str, must_exist: bool = False):
    super(DatatypeDirectory, self).__init__(url, Base, must_exist = must_exist)
    return
