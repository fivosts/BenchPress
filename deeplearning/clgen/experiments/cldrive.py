"""
Scratchpad experimental analysis for anyhing related to CLDrive.
"""
import pathlib
import pandas as pd
import statistics
import numpy as np
import scipy.stats as st

from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import plotter as plt

def confidenceInterval():
  """
  For a simple and a more complicated OpenCL kernel,
  this experiment uses a fixed local size, iterates over a wide range of global sizes
  and a different number of runs to check the stability of execution times.

  The 95% confidence interval is calculated for each run and the average distance of the mean
  from the confidence interval boundaries is expressed as a percentage distance.
  """
  src1 ="""
  kernel void A(global float *a, global float* n, global float* j, global float* h, const unsigned int r) {
    int l = get_global_id(0);
    if (l >= r)
      return;
    h[l] = n[l] - j[l]  * j[l-1];
  }

  """
  src2 ="""
  kernel void A(global float *a, global float* n, global float* j, global float* h, const unsigned int r) {
    int l = get_global_id(0);
    for (unsigned int i = 0; i < 10; i++){
      a[i] = n[i]*j[i]*h[i];
      a[i+1] = n[i+1]*j[i+1]*h[i+1];
    }
  }
  """

  MIN_GISZE_POW = 8
  MAX_GSIZE_POW = 28
  MAX_RUNS_POW  = 5

  for n, src in [("src1", src1), ("src2", src2)]:
    for gsize_pow in range(MIN_GISZE_POW, MAX_GSIZE_POW+1):
      print("Running {}, {} gsize".format(gsize_pow, 2**gsize_pow))
      data_cpt = []
      data_cpk = []
      data_gpt = []
      data_gpk = []
      for n_runs in range(1, MAX_RUNS_POW+1):
        print("##### num_runs: {} ####".format(10**n_runs))
        hist = {}
        cpt, cpk, gpt, gpk = opencl.CLDriveExecutionTimes(src, num_runs = 10**n_runs, gsize = 2**gsize_pow, lsize = 32)

        print("## CPU transfer")
        interval = st.t.interval(alpha = 0.95, df = len(list(cpt)) -1, loc = np.mean(list(cpt)), scale = st.sem(list(cpt)))
        cpt_mean = cpt.mean()
        ratio = 100*(0.5 * (interval[1] - interval[0])) / cpt_mean
        data_cpt.append(ratio)
        print("95 interval: {}".format(interval))
        print("Ratio: {}%".format(100*(0.5 * (interval[1] - interval[0])) / cpt_mean))

        print("## CPU kernel")
        interval = st.t.interval(alpha = 0.95, df = len(list(cpk)) -1, loc = np.mean(list(cpk)), scale = st.sem(list(cpk)))
        cpk_mean = cpk.mean()
        ratio = 100*(0.5 * (interval[1] - interval[0])) / cpk_mean
        data_cpk.append(ratio)
        print("95 interval: {}".format(interval))
        print("Ratio: {}%".format(100*(0.5 * (interval[1] - interval[0])) / cpk_mean))

        print("## GPU transfer")
        interval = st.t.interval(alpha = 0.95, df = len(list(gpt)) -1, loc = np.mean(list(gpt)), scale = st.sem(list(gpt)))
        gpt_mean = gpt.mean()
        ratio = 100*(0.5 * (interval[1] - interval[0])) / gpt_mean
        data_gpt.append(ratio)
        print("95 interval: {}".format(interval))
        print("Ratio: {}%".format(100*(0.5 * (interval[1] - interval[0])) / gpt_mean))

        print("## GPU kernel")
        interval = st.t.interval(alpha = 0.95, df = len(list(gpk)) -1, loc = np.mean(list(gpk)), scale = st.sem(list(gpk)))
        gpk_mean = gpk.mean()
        ratio = 100*(0.5 * (interval[1] - interval[0])) / gpk_mean
        data_gpk.append(ratio)
        print("95 interval: {}".format(interval))
        print("Ratio: {}%".format(100*(0.5 * (interval[1] - interval[0])) / gpk_mean))
        print()
      x_axis = [10**x for x in range(1, MAX_RUNS_POW+1)]
      plt.MultiScatterLine(
        x = [x_axis]*MAX_RUNS_POW,
        y = [data_cpt, data_cpk, data_gpt, data_gpk],
        names = ['cpu_transfer', 'cpu_kernel', 'gpu_transfer', 'gpu_kernel'],
        plot_name = "{}_perc_diff_mean_int".format(2**gsize_pow),
        path = pathlib.Path("./plots/{}".format(n)).resolve()
      )
  return
