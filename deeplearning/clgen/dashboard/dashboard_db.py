import datetime
import os

import sqlalchemy as sql
from sqlalchemy.dialects import mysql

from absl import flags
from deeplearning.clgen.util import sqlutil

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS
Base = sqlutil.Base()

class DashboardDatabase(sqlutil.Database):
  def __init__(self, url: str, must_exist: bool):
    super(DashboardDatabase, self).__init__(url, Base, must_exist=must_exist)

class Corpus(Base):
  __tablename__ = "corpuses"

  id: int = sql.Column(sql.Integer, primary_key=True)
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )
  preprocessed_url: str = sql.Column(sql.String(256), nullable=False)
  encoded_url: str = sql.Column(sql.String(256), nullable=False)
  summary: str = sql.Column(sql.String(256), nullable=False)

  __table_args__ = (
    sql.UniqueConstraint(
      "config_proto_sha1",
      "preprocessed_url",
      "encoded_url",
      name="unique_corpus",
    ),
  )

class Model(Base):
  __tablename__ = "models"

  id: int = sql.Column(sql.Integer, primary_key=True)
  corpus_id: int = sql.Column(
    sql.Integer, sql.ForeignKey("corpuses.id"), nullable=False,
  )
  config_proto_sha1: str = sql.Column(sql.String(40), nullable=False)
  config_proto: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )
  cache_path: str = sql.Column(sql.String(256), nullable=False)
  summary: str = sql.Column(sql.String(256), nullable=False)

  corpus: Corpus = sql.orm.relationship("Corpus")
  __table_args__ = (
    sql.UniqueConstraint(
      "corpus_id", "config_proto_sha1", "cache_path", name="unique_model"
    ),
  )

class TrainingTelemetry(Base):
  __tablename__ = "training_telemetry"

  id: int = sql.Column(sql.Integer, primary_key=True)
  model_id: int = sql.Column(
    sql.Integer, sql.ForeignKey("models.id"), nullable=False,
  )
  timestamp: datetime.datetime = sql.Column(
    sql.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql"),
    nullable=False,
    default=lambda x: datetime.datetime.utcnow(),
  )
  epoch: int = sql.Column(sql.Integer, nullable=False)
  step: int = sql.Column(sql.Integer, nullable=False)
  training_loss: float = sql.Column(sql.Float, nullable=False)
  learning_rate: float = sql.Column(sql.Float, nullable=False)
  ns_per_batch: int = sql.Column(sql.Integer, nullable=False)

  pending: bool = sql.Column(sql.Boolean, nullable=False, default=True)

  model: Model = sql.orm.relationship("Model")
  __table_args__ = (
    sql.UniqueConstraint("model_id", "epoch", "step", name="unique_telemetry"),
  )

class TrainingSample(Base):
  __tablename__ = "training_samples"

  id: int = sql.Column(sql.Integer, primary_key=True)
  model_id: int = sql.Column(
    sql.Integer, sql.ForeignKey("models.id"), nullable=False,
  )
  epoch: int = sql.Column(sql.Integer, nullable=False)
  step: int = sql.Column(sql.Integer, nullable=False)
  token_count: int = sql.Column(sql.Integer, nullable=False)
  sample_time: int = sql.Column(sql.Integer, nullable=False)
  sample: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )

  model: Model = sql.orm.relationship("Model")

def GetDatabase() -> DashboardDatabase:
  db: DashboardDatabase = DashboardDatabase(
          url = "sqlite:///{}/dashboard.db".format(os.path.abspath(FLAGS.workspace_dir)), must_exist = False
          )
  l.logger().info("Created dashboard database {}".format(db.url))
  return db
