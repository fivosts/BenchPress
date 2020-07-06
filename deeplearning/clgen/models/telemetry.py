"""This file defines telemetry data gathers."""
import pathlib
import re
import typing
import datetime
from absl import flags

from deeplearning.clgen.proto import telemetry_pb2
from deeplearning.clgen.util import pbutil
from eupy.native import logger as l

FLAGS = flags.FLAGS


class TrainingLogger(object):
  """A TrainingLogger produces telemetry data of a CLgen model as it is trained.

  Telemetry data is gathered after every epoch of training. It includes a
  timestamp, the model's loss, and the time spent training the epoch.

  See the Keras callback docs: https://keras.io/callbacks/#lambdacallback
  """

  def __init__(self, logdir: pathlib.Path):
    logdir.mkdir(exist_ok = True)
    self.logdir = logdir
    self.last_epoch_begin_timestamp = None
    self.telemetry = None

  def EpochBeginCallback(self) -> None:
    self.last_epoch_begin_timestamp = datetime.datetime.utcnow()

  def EpochEndCallback(self, epoch: int, loss: float):
    now = datetime.datetime.utcnow()
    epoch_time_ms = now - self.last_epoch_begin_timestamp
    telemetry = telemetry_pb2.ModelEpochTelemetry(
      timestamp_unix_epoch_ms = now.strftime("%m/%d/%Y, %H:%M:%S"),
      epoch_num = epoch,
      epoch_wall_time_ms = int(round(epoch_time_ms.total_seconds()*1000)),
      loss = loss,
    )
    pbutil.ToFile(telemetry, self.logdir / f"epoch_{epoch:03d}_telemetry.pbtxt")

  def KerasEpochBeginCallback(self, epoch: int, logs: typing.Union[typing.List[typing.Any], typing.Dict[str, typing.Any]]) -> None:
    """A Keras "on_epoch_end" callback."""
    del epoch
    del logs
    self.EpochBeginCallback()

  def KerasEpochEndCallback(self, epoch: int, logs: typing.Union[typing.List[typing.Any], typing.Dict[str, typing.Any]]) -> None:
    """A Keras "on_epoch_end" callback."""
    # Keras epoch numbers are zero indexed.
    self.EpochEndCallback(epoch + 1, logs["loss"])

  def KerasCallback(self, keras):
    """Returns the keras callback to passed to a model's fit() function."""
    return keras.callbacks.LambdaCallback(
      on_epoch_begin=self.KerasEpochBeginCallback,
      on_epoch_end=self.KerasEpochEndCallback,
    )

  def TfRecordEpochs(self) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(str(self.logdir))
    event_acc.Reload()
    try:
      wall_time, step_nums, loss = zip(*event_acc.Scalars('training/total_loss'))
    except KeyError as e:
      l.getLogger().warn("Model loss log not found! Available Tags: {}".format(event_acc.Tags()))
      self.telemetry = [
        telemetry_pb2.ModelEpochTelemetry(
          timestamp_unix_epoch_ms = str(0),
          epoch_num = 0,
          epoch_wall_time_ms = 0,
          loss = -1,
        )
      ]
      return
      
    assert len(wall_time) == len(step_nums)
    assert len(step_nums) == len(loss)

    self.telemetry = []
    for (indx, (wt, ls)) in enumerate(zip(wall_time, loss)):
      round_wt = int(round(wt, 0))
      if indx == 0:
        current_time = round_wt
        continue
      else:
        self.telemetry.append(telemetry_pb2.ModelEpochTelemetry(
                                  timestamp_unix_epoch_ms = str(round_wt),
                                  epoch_num = indx,
                                  epoch_wall_time_ms = round_wt - current_time,
                                  loss = ls,
                              )
          )
        current_time = round_wt

  def EpochTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Return the epoch telemetry files."""
    if self.telemetry is None:
      return [
        pbutil.FromFile(self.logdir / p, telemetry_pb2.ModelEpochTelemetry())
        for p in sorted(self.logdir.iterdir())
        if re.match(r"epoch_\d\d+_telemetry\.pbtxt", str(p.name))
      ]
    else:
      return self.telemetry
