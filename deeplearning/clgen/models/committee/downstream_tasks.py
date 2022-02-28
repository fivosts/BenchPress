"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""

TASKS = {
  "GrewePredictive": GrewePredictive,
}

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str) -> "DownstreamTask":
    return TASKS[task]()

  def __init__(self, name) -> None:
    self.name = name
    return

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  """
  def __init__(self) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive")
    self.inputs        = ["comp", "mem", "localmem", "coalesced", "atomic"]
    self.input_size    = 10
    self.output_labels = ["CPU", "GPU"]
    self.output_size   = 2
    return