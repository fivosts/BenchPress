"""
Target benchmark analysis evaluator.
"""
def AnalyzeTarget(**kwargs) -> None:
  """
  Analyze requested target benchmark suites.
  """
  targets   = kwargs.get('targets')
  tokenizer = kwargs.get('tokenizer')
  workspace_path = kwargs.get('workspace_path')
  raise NotImplementedError
  return