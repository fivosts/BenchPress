"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""

TASKS = {
  "GrewePredictive": GrewePredictive,
}

class DownstreamTask(object):
  @classmethod
  def FromTask(cls, task: str) -> "DownstreamTask":
    return TASKS[task]()

  def __init__(self, task) -> None:
    return

class GrewePredictive(DownstreamTask):
  def __init__(self) -> None:
    super(GrewePredictive, self).__init__()
    return