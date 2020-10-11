"""Github mining configuration"""
from deeplearning.clgen.proto import github_miner_pb2
from absl import flags
from deeplearning.clgen.util import pbutil

FLAGS = flags.FLAGS

def AssertConfig(config: github_miner_pb2.GitHubMiner) -> None:
  """Evaluates correct usage of github miner protobuf fields."""
  try:
    pbutil.AssertFieldIsSet(config, "path")
    pbutil.AssertFieldIsSet(config, "data_format")
    pbutil.AssertFieldIsSet(config, "miner")

    if config.miner.HasField("big_query"):
      pbutil.AssertFieldIsSet(config.miner.big_query, "credentials")
      pbutil.AssertFieldConstraint(
        config.miner.big_query,
        "language",
        lambda x: x in {'generic', 'opencl', 'c', 'cpp', 'java', 'python'},
        "language must be one of opencl, c, cpp, java, python. 'generic' for language agnostic queries.",
      )
      miner = BigQuery(config)
    elif config.miner.HasField("recursive"):
      pbutil.AssertFieldConstraint(
        config.miner.recursive,
        "flush_limit_K",
        lambda x: x>0,
        "flush limit cannot be non-positive."
        )
      pbutil.AssertFieldConstraint(
        config.miner.recursive,
        "corpus_size_K",
        lambda x: x>0,
        "corpus size cannot be non-positive."
        )
    else:
      raise SystemError("{} miner not recognized".format(config.miner))
  except Exception as e:
    raise e
  return


class GithubMiner(object):
  def __init__(self):
    return

  @classmethod
  def FromConfig(config: github_miner_pb2.GitHubMiner) -> GitHubMiner:
    """Constructs github miner from protobuf configuration."""
    AssertConfig(config)
    return

