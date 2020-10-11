"""Github mining configuration"""
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.proto import github_miner_pb2
from deeplearning.clgen.corpuses.github import big_query
from deeplearning.clgen.corpuses.github import recursive

class GithubMiner(object):
  """Base abstract class of a github miner"""

  @classmethod
  def FromConfig(cls, config: github_miner_pb2.GitHubMiner) -> GitHubMiner:
    """Constructs github miner from protobuf configuration."""
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
        return big_query.BigQuery(config)
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
        if config.data_format != config.GitHubMiner.DataFormat.folder:
          raise NotImplementedError("RecursiveFetcher only stores files in local folder.")
        return recursive.RecursiveFetcher(config)
      else:
        raise SystemError("{} miner not recognized".format(config.miner))
    except Exception as e:
      raise e

  def __init__(self):
    return

  def fetch(self) -> None:
    raise NotImplementedError("Abstract class")