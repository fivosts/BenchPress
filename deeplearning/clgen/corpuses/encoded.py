# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file defines a database for encoded content files."""
import datetime
import functools
import multiprocessing
import pickle
import time
import typing
import pathlib

import numpy as np
import progressbar
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func


from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.util import monitors
from deeplearning.clgen.features import extractor
from absl import flags
import humanize
from deeplearning.clgen.util import sqlutil

from eupy.native import logger as l

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

flags.DEFINE_boolean(
  "override_encoding",
  False,
  "Set to override incomplete encoding. Does not set DB value to 'done'"
)

class Meta(Base):
  """Meta table for encoded content files database."""

  __tablename__ = "meta"

  key: str = sql.Column(sql.String(1024), primary_key = True)
  value: str = sql.Column(sql.String(1024), nullable = False)

class EncodedContentFileStats(Base):
  """Stats table for encoded content files."""

  __tablename__ = "encoded_contentfiles_stats"

  # Total number of files.
  file_count       : int = sql.Column(sql.Integer,      primary_key = True)
  # Average feature vector of contentfiles.
  corpus_features  : str = sql.Column(sql.String(1024), nullable = False)
  # Token length distribution of contentfiles.
  corpus_lengths   : str = sql.Column(sql.String(1024), nullable = False)

class EncodedContentFile(Base):
  """A single encoded content file."""

  __tablename__ = "encoded_contentfiles"

  # The ID of the PreprocessedContentFile.
  id: int = sql.Column(sql.Integer, primary_key=True)
  # We store the vocabulary indices array as a string of period-separated
  # integers, e.g. '0.1.2.0.1'. To access the values as an array of integers,
  # use EncodedContentFile.indices_array.
  data: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)
  # Number of tokens in sequence
  tokencount: int = sql.Column(sql.Integer, nullable=False)
  # Sequence features extracted.
  feature_vector: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # The number of milliseconds encoding took.
  encoding_time_ms: int = sql.Column(sql.Integer, nullable=False)
  # Encoding is parallelizable, so the actual wall time of encoding may be much
  # less than the sum of all encoding_time_ms. This column counts the effective
  # number of "real" milliseconds during encoding between the last encoded
  # result and this result coming in. The idea is that summing this column
  # provides an accurate total of the actual time spent encoding an entire
  # corpus. Will be <= encoding_time_ms.
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @staticmethod
  def DataStringToNumpyArray(data: str) -> np.ndarray:
    """Convert the 'data' string to a numpy array."""
    return np.array([int(x) for x in data.split(".")], dtype=np.int32)

  @staticmethod
  def NumpyArrayToDataString(array: np.ndarray) -> str:
    """Convert the 'data' string to a numpy array."""
    return ".".join(str(x) for x in array)

  @property
  def indices_array(self) -> np.ndarray:
    """The numpy array of the encoded data."""
    return self.DataStringToNumpyArray(self.data)

  @property
  def features(self) -> typing.Dict[str, float]:
    return extractor.RawToDictFeats(self.feature_vector)

  @classmethod
  def FromPreprocessed(
    cls,
    preprocessed_cf: preprocessed.PreprocessedContentFile,
    tokenizer: tokenizers.TokenizerBase,
    eof: str,
    pre_train: bool,
  ) -> "EncodedContentFile":
    """Instantiate an EncodedContentFile from a preprocessed file.

    Args:
      preprocessed_cf: A PreprocessedContentFile instance.
      tokenizer: The tokenizer to encode using.
      eof: An end-of-file marker which is concatenated to the encoded sequence.

    Returns:
      An EncodedContentFile instance.
    """
    start_time = time.time()
    data = tokenizer.TokenizeString(preprocessed_cf.text)
    ####
    # TODO kernel analytics
    # encoded_length = len(data)
    # token_values = data.sorted()
    ####
    encoding_time_ms = int((time.time() - start_time) * 1000)
    try:
      if not pre_train:
        feature_vector = extractor.ExtractRawFeatures(preprocessed_cf.text)
      else:
        feature_vector = ""
    except Exception as e:
      raise e
    return EncodedContentFile(
      id = preprocessed_cf.id,
      # Encode the end-of-file marker separately to ensure that it resolves to
      # the correct token. For example if the vocabulary contains 'a', 'b',
      # and 'ab', then a content file 'a' with EOF marker 'b' would be encoded
      # as 'ab', instead of 'a'+'b'.
      data = cls.NumpyArrayToDataString(
        np.concatenate((data, tokenizer.TokenizeString(eof)))
      ),
      tokencount       = len(data),
      feature_vector   = feature_vector,
      encoding_time_ms = encoding_time_ms,
      wall_time_ms     = encoding_time_ms,  # The outer-loop may change this.
      date_added       = datetime.datetime.utcnow(),
    )


def EncoderWorker(
  job: internal_pb2.EncoderWorker,
  tokenizer,
  contentfile_separator,
  is_pre_train,
) -> typing.Optional[EncodedContentFile]:
  """Encode a single content file."""
  # TODO(cec): There is a bug in the tokenizer creation logic such that the
  # derived tokenizer is not always capable of encoding the preprocessed files.
  # Once this has been fixed, there is no need to catch the VocabError here,
  # and EncoderWorker can always return an EncodedContentFile instance.
  try:
    return EncodedContentFile.FromPreprocessed(
      preprocessed.PreprocessedContentFile(id=job.id, text=job.text),
      tokenizer,
      contentfile_separator,
      is_pre_train,
    )
  except Exception as e:
    raise e


class EncodedContentFiles(sqlutil.Database):
  """A database of encoded pre-processed contentfiles."""

  def __init__(self, url: str, is_pre_train: bool = False, must_exist: bool = False):
    self.encoded_path = pathlib.Path(url.replace("sqlite:///", "")).parent
    self.is_pre_train     = is_pre_train
    self.length_monitor   = monitors.CumulativeHistMonitor(self.encoded_path, "encoded_kernel_length")
    if not self.is_pre_train:
      self.token_monitor    = monitors.NormalizedFrequencyMonitor(self.encoded_path, "token_distribution")
      self.feature_monitors = {ftype: monitors.CategoricalDistribMonitor(self.encoded_path, "{}_distribution".format(ftype)) for ftype in extractor.extractors.keys()}
    super(EncodedContentFiles, self).__init__(url, Base, must_exist=must_exist)

  def Create(
    self,
    p: preprocessed.PreprocessedContentFiles,
    tokenizer: tokenizers.TokenizerBase,
    contentfile_separator: str,
  ) -> bool:
    """Populate the encoded contentfiles database.

    Args:
      p: A PreprocessedContentFiles database.
      tokenizer: An TokenizerBase instance.
      contentfile_separator: The contentfile separator.

    Returns:
      True if work was done, else False.

    Raises:
      EmptyCorpusException: If the PreprocessedContentFiles database has
        no files.
    """
    with self.Session() as session:
      if not self.IsDone(session):
        self.Import(session, p, tokenizer, contentfile_separator)
        self.SetStats(session)
        self.SetDone(session)
        session.commit()

      # Logging output.
    #   num_files = session.query(EncodedContentFile).count()
    #   token_count, total_walltime, total_time, = session.query(
    #     func.sum(EncodedContentFile.tokencount),
    #     func.sum(EncodedContentFile.wall_time_ms),
    #     func.sum(EncodedContentFile.encoding_time_ms),
    #   ).first()
    # l.getLogger().info("Encoded {} files in {} ms ({:.2f}x speedup)"
    #                     .format(
    #                         humanize.intcomma(num_files),
    #                         humanize.intcomma(total_walltime),
    #                         total_time / total_walltime,
    #                       ), mail_level = 4
    #                   )
    # l.getLogger().info("Encoded corpus: {} tokens, {} files."
    #                     .format(
    #                         humanize.intcomma(token_count),
    #                         humanize.intcomma(num_files),
    #                       ), mail_level = 4
    #                   )
    return

  @property
  def size(self):
    """Return the total number of files in the encoded corpus."""
    with self.Session() as session:
      return session.query(EncodedContentFile).count()

  @property
  def token_count(self) -> int:
    """Return the total number of tokens in the encoded corpus.

    This excludes the EOF markers which are appended to each encoded text.
    """
    with self.Session() as session:
      return session.query(func.sum(EncodedContentFile.tokencount)).scalar()

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == "done").first():
      return True
    elif FLAGS.override_encoding:
      l.getLogger().warn("Overriding incomplete encoded DB.")
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key="done", value="yes"))

  def SetStats(self, session: sqlutil.Session) -> None:
    """Write corpus stats to DB"""
    file_count      = session.query(EncodedContentFile.id).count()
    if not self.is_pre_train:
      corpus_features = '\n\n'.join([ftype + ":\n" + mon.getStrData() for ftype, mon in self.feature_monitors.items()])
    else:
      corpus_features = ""
    corpus_lengths  = self.length_monitor.getStrData()

    if session.query(EncodedContentFileStats).first():
      stats = session.query(EncodedContentFileStats).first()
      stats.file_count      = file_count
      stats.corpus_features = corpus_features
      stats.corpus_lengths  = corpus_lengths
    else:
      session.add(
        EncodedContentFileStats(
          file_count = file_count,
          corpus_features = corpus_features,
          corpus_lengths = corpus_lengths,
        )
      )
    return

  def Import(
    self,
    session: sqlutil.Session,
    preprocessed_db: preprocessed.PreprocessedContentFiles,
    tokenizer: tokenizers.TokenizerBase,
    contentfile_separator: str,
  ) -> None:
    with preprocessed_db.Session() as p_session:
      query = p_session.query(preprocessed.PreprocessedContentFile).filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True,
        ~preprocessed.PreprocessedContentFile.id.in_(
          session.query(EncodedContentFile.id).all()
        ),
      )
      # jobs = [
      #   internal_pb2.EncoderWorker(
      #     id=x.id,
      #     text=x.text,
      #     contentfile_separator=contentfile_separator,
      #     # pickled_tokenizer=pickle.dumps(tokenizer),
      #   )
      #   for x in query
      # ]
      # if not jobs:
      #   raise ValueError(
      #     "Pre-processed corpus contains no files: " f"'{preprocessed_db.url}'"
      #   )
      total_jobs = query.count()
      l.getLogger().info("Encoding {} of {} preprocessed files"
                          .format(
                              humanize.intcomma(total_jobs),
                              humanize.intcomma(
                                p_session.query(preprocessed.PreprocessedContentFile)
                                .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True)
                                .count()
                              )
                          )
                        )
      bar = progressbar.ProgressBar(max_value=total_jobs)
      chunk, idx = 2000000, 0
      last_commit = time.time()
      wall_time_start = time.time()
      while idx < total_jobs:
        try:
          batch = query.limit(chunk).offset(idx).all()
          pool = multiprocessing.Pool()
          for encoded_cf in pool.imap_unordered(
                              functools.partial(EncoderWorker,
                                                tokenizer = tokenizer,
                                                contentfile_separator = contentfile_separator,
                                                is_pre_train = self.is_pre_train,
                                                ),
                              batch
                            ):
            wall_time_end = time.time()
            # TODO(cec): Remove the if check once EncoderWorker no longer returns
            # None on tokenizer encode error.
            if encoded_cf:
              encoded_cf.wall_time_ms = int(
                (wall_time_end - wall_time_start) * 1000
              )
              session.add(encoded_cf)
              self.length_monitor.register(encoded_cf.tokencount)
              if not self.is_pre_train:
                self.token_monitor.register([tokenizer.decoder[int(x)] for x in encoded_cf.data.split('.')])

                dict_features = extractor.RawToDictFeats(encoded_cf.feature_vector)
                if dict_features:
                  for key, value in dict_features.items():
                    self.feature_monitors[key].register(value)
            wall_time_start = wall_time_end
            if wall_time_end - last_commit > 10:
              session.commit()
              last_commit = wall_time_end
            idx += 1
            bar.update(idx)
          pool.close()
        except KeyboardInterrupt as e:
          pool.terminate()
          self.length_monitor.plot()
          if not self.is_pre_train:
            self.token_monitor.plot()
            for m in self.feature_monitors.values():
              m.plot()
          raise e
        except Exception as e:
          l.getLogger().error(e)
          pool.terminate()
          self.length_monitor.plot()
          if not self.is_pre_train:
            self.token_monitor.plot()
            for m in self.feature_monitors.values():
              m.plot()
          raise e
      self.length_monitor.plot()
      if not self.is_pre_train:
        self.token_monitor.plot()
        for m in self.feature_monitors.values():
          m.plot()
    session.commit()
    return

  @staticmethod
  def GetVocabFromMetaTable(session) -> typing.Dict[str, int]:
    """Read a vocabulary dictionary from the 'Meta' table of a database."""
    q = session.query(Meta.value).filter(Meta.key == "vocab_size")
    if not q.first():
      return {}

    vocab_size = int(q.one()[0])
    q = session.query(Meta.value)
    return {
      q.filter(Meta.key == f"vocab_{i}").one()[0]: i for i in range(vocab_size)
    }

  @staticmethod
  def StoreVocabInMetaTable(
    session: sqlutil.Session, vocabulary: typing.Dict[str, int]
  ) -> None:
    """Store a vocabulary dictionary in the 'Meta' table of a database."""
    q = session.query(encoded.Meta).filter(encoded.Meta.key.like("vocab_%"))
    q.delete(synchronize_session=False)

    session.add(encoded.Meta(key="vocab_size", value=str(len(vocabulary))))
    session.add_all(
      [encoded.Meta(key=f"vocab_{v}", value=k) for k, v in vocabulary.items()]
    )

  def get_data(self, sequence_length: int = None) -> typing.List[np.array]:
    """
    Get the indices array of encoded contentfiles.
    """
    with self.Session() as session:
      if sequence_length:
        return [x.indices_array for x in session.query(EncodedContentFile).filter(EncodedContentFile.tokencount <= sequence_length).all()]
      else:
        return [x.indices_array for x in session.query(EncodedContentFile).all()]

  def get_features(self, sequence_length: int = None) -> typing.List[str]:
    """
    Get feature vectors of training instances within the specified sequence length.
    """
    with self.Session() as session:
      if sequence_length:
        return [x.feature_vector for x in session.query(EncodedContentFile).filter(EncodedContentFile.tokencount <= sequence_length).all()]
      else:
        return [x.feature_vector for x in session.query(EncodedContentFile).all()]
