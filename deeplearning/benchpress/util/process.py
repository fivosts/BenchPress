import multiprocessing

def isolate(process: callable, **kwargs) -> None:
  """
  Executes a callable in isolated process space by spawning a child process.
  After executing, memory, cpu and gpu resources will be freed.

  Handy in executing TF-graph functions that will not free memory after execution.
  Args:
    process: callable. Function to be executed.
    kwargs: See multiprocessing.Process docs for kwargs.
  """
  pr = multiprocessing.Process(target = process, **kwargs)
  pr.start()
  pr.join()
  return