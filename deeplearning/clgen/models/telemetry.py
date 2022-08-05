"""This file defines telemetry data gathers."""
import pathlib
import re
import json
import typing
import datetime
import glob
from absl import flags

from deeplearning.benchpress.proto import telemetry_pb2
from deeplearning.benchpress.util import pbutil
from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS


class TrainingLogger(object):
  """A TrainingLogger produces telemetry data of a CLgen model as it is trained.

  Telemetry data is gathered after every epoch of training. It includes a
  timestamp, the model's loss, and the time spent training the epoch.

  See the Keras callback docs: https://keras.io/callbacks/#lambdacallback
  """

  def __init__(self, logdir: pathlib.Path):
    logdir.mkdir(exist_ok = True, parents = True)
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
    self.tfAccumulateLoss(event_acc)

    for key in event_acc.Tags()['scalars']:
      _, step, value = zip(*event_acc.Scalars(key))
      key_str = str(pathlib.Path(key).stem)
      plt.linesSingleAxis(
        {key_str: {'y': value, 'x': step}},
        y_name = key_str,
        x_name = "Train step",
        plot_title = key_str,
        path       = self.logdir,
      )
    return

  def tfAccumulateLoss(self, event_acc):
    """Open accumulator and read total_loss scalar"""
    try:
      self.telemetry = []
      wall_time, step_num, loss = zip(*event_acc.Scalars('training/total_loss'))
      for (indx, (wt, st, ls)) in enumerate(zip(wall_time, step_num, loss)):
        round_wt = int(round(wt, 0))
        if indx == 0:
          current_time = round_wt
          continue
        else:
          self.telemetry.append(telemetry_pb2.ModelEpochTelemetry(
                                    timestamp_unix_epoch_ms = str(round_wt),
                                    epoch_num = st,
                                    epoch_wall_time_ms = round_wt - current_time,
                                    loss = ls,
                                )
            )
          current_time = round_wt
    except KeyError as e:
      l.logger().warn("Model loss log not found! Available Tags: {}".format(event_acc.Tags()))
      self.telemetry = [
        telemetry_pb2.ModelEpochTelemetry(
          timestamp_unix_epoch_ms = str(0),
          epoch_num = 0,
          epoch_wall_time_ms = 0,
          loss = -1,
        )
      ]
    return

  def EpochTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Return the epoch telemetry files."""
    if self.telemetry is None:
      if len(glob.glob(str(self.logdir / "epoch_*_telemetry.pbtxt"))) > 0:
        return [
          pbutil.FromFile(self.logdir / p, telemetry_pb2.ModelEpochTelemetry())
          for p in sorted(self.logdir.iterdir())
          if re.match(r"epoch_\d\d+_telemetry\.pbtxt", str(p.name))
        ]
      elif len(glob.glob(str(self.logdir / "events.out.tfevents*"))) > 0:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_acc = EventAccumulator(str(self.logdir))
        event_acc.Reload()
        self.tfAccumulateLoss(event_acc)
      elif len(glob.glob(str(self.logdir / "training.json"))) == 1:
        with open(self.logdir / "training.json", 'r') as jsf:
          data = json.load(jsf)
        self.telemetry = [
          telemetry_pb2.ModelEpochTelemetry(
            timestamp_unix_epoch_ms = '0',
            epoch_num = x['step'],
            epoch_wall_time_ms = int(round(x['batch_execution_time_ms'])) if "batch_execution_time_ms" in x else -1,
            loss = x['total_loss'] if "total_loss" in x else -1.0,
          ) for x in data
        ]
      else:
        l.logger().warn("Training logs have not been found. Invalid reported loss.")
        self.telemetry = [
          telemetry_pb2.ModelEpochTelemetry(
            timestamp_unix_epoch_ms = str(0),
            epoch_num = 0,
            epoch_wall_time_ms = 0,
            loss = -1,
          )
        ]
    return self.telemetry
