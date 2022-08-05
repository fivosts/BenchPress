import pathlib
import subprocess
import time
import typing
import os
from deeplearning.benchpress.preprocessors import opencl, clang
from deeplearning.benchpress.corpuses import encoded
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.features import autophase
from deeplearning.benchpress.util import logging as l
from deeplearning.benchpress.util import distributions

from absl import app

def main(*args):

  db = encoded.EncodedContentFiles("sqlite:///{}".format(pathlib.Path("./unique_encoded.db").resolve()), must_exist = True)
  tokenizer = tokenizers.TokenizerBase.FromFile(pathlib.Path("./backup_tokenizer.pkl").resolve())
  data = [tokenizer.ArrayToCode(x) for x in db.get_data()][:30]

  args = builtin_cflags = opencl.GetClangArgs(use_shim = False, use_aux_headers = False) + ["-S", "-emit-llvm", "-o", "-"]

  stdin_times, opencl_times, bytecode_times = [], [], []

  opt_stdin, opt_file = [], []

  for idx, src in enumerate(data):

    print(idx)

    # for x in range(50):
    #   t1 = time.time()
    #   opencl.CompileStdin(src)
    #   t2 = time.time()
    #   stdin_times.append(int(1000 * (t2-t1)))


    # for x in range(50):
    #   t1 = time.time()
    #   opencl.Compile(src)
    #   t2 = time.time()
    #   opencl_times.append(int(1000 * (t2 - t1)))

    # for x in range(50):
    #   t1 = time.time()
    #   opencl.CompileLlvmBytecode(src)
    #   t2 = time.time()
    #   bytecode_times.append(int(1000 * (t2 - t1)))

    for x in range(100):
      t1 = time.time()
      opencl.CompileOptimizer(src, autophase.AUTOPHASE)
      t2 = time.time()
      opt_file.append(int(1000 * (t2 - t1)))

    for x in range(100):
      t1 = time.time()
      opencl.CompileOptimizerStdin(src, autophase.AUTOPHASE)
      t2 = time.time()
      opt_stdin.append(int(1000 * (t2 - t1)))

  # stdin_distr    = distributions.GenericDistribution(stdin_times, "process_benchmarks", "stdin")
  # opencl_distr   = distributions.GenericDistribution(opencl_times, "process_benchmarks", "opencl")
  # bytecode_distr = distributions.GenericDistribution(bytecode_times, "process_benchmarks", "bytecode")

  opt_stdin_distr = distributions.GenericDistribution(opt_stdin, "process_benchmarks", "opt_stdin")
  opt_file_distr  = distributions.GenericDistribution(opt_file, "process_benchmarks", "opt_file")


  # stdin_distr.plot()
  # opencl_distr.plot()
  # bytecode_distr.plot()

  opt_stdin_distr.plot()
  opt_file_distr.plot()

  # cum = stdin_distr - opencl_distr
  # cum2 = stdin_distr - bytecode_distr
  cum3 = opt_stdin_distr - opt_file_distr

  # cum.plot()
  # cum2.plot()
  cum3.plot()

  # print(cum < 0)
  # print(cum2 < 0)
  print(cum3 < 0)

  print(opt_stdin_distr.average, opt_stdin_distr.median)
  print(opt_file_distr.average, opt_file_distr.median)
  print(cum3.average, cum3.median)

  print()

if __name__ == "__main__":
  app.run(main)
