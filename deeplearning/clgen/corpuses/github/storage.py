"""BigQuery Dataset structures"""
import os
import typing
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

class bqStorage(object):

  @classmethod
  def FromArgs(cls, ):
    return

  def __init__(self):
    return

class zipStorage(bqStorage):
  def __init__(self):
    super(zipStorage, self).__init__()

class fileStorage(bqStorage):
  def __init__(self):
    super(fileStorage, self).__init__()

class dbStorage(bqStorage):
  def __init__(self):
    super(dbStorage, self).__init__()

class bqTableStorage(bqStorage):
  def __init__(self):
    super(bqTableStorage, self).__init__()