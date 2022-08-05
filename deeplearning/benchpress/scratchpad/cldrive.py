"""
Scratchpad experimental analysis for anyhing related to CLDrive.
"""
import pathlib
import math
import pandas as pd
import statistics
import numpy as np
import scipy.stats as st

from deeplearning.benchpress.preprocessors import opencl
from deeplearning.benchpress.util import plotter as plt

from deeplearning.benchpress.util import logging as l

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

l.initLogger(name = "experiments")

def confidenceInterval() -> None:
  """
  For a simple and a more complicated OpenCL kernel,
  this experiment uses a fixed local size, iterates over a wide range of global sizes
  and a different number of runs to check the stability of execution times.

  The 95% confidence interval is calculated for each run and the average distance of the mean
  from the confidence interval boundaries is expressed as a percentage distance.
  """

  global src1
  global src2

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
      x_axis = [x for x in range(1, MAX_RUNS_POW+1)]
      plt.MultiScatterLine(
        x = [x_axis] * MAX_RUNS_POW,
        y = [data_cpt, data_cpk, data_gpt, data_gpk],
        names = ['cpu_transfer', 'cpu_kernel', 'gpu_transfer', 'gpu_kernel'],
        x_name = "power of 10",
        plot_name = "{}_perc_diff_mean_int".format(2**gsize_pow),
        path = pathlib.Path("./plots/conf_interval/{}".format(n)).resolve()
      )
  return

def LabelGlobalSize() -> None:
  """
  Iterate over multiple global sizes and collect the optimal device
  to execute an OpenCL kernel. GPU or CPU.
  """
  global src1
  global src2

  MIN_GISZE_POW = 8
  MAX_GSIZE_POW = 28
  N_RUNS = {
    'src1': {
      2**8 : 10**5,
      2**9 : 10**5,
      2**10: 10**5,
      2**11: 10**5,
      2**12: 10**5,
      2**13: 10**5,
      2**14: 10**4,
      2**15: 10**4,
      2**16: 10**4,
      2**17: 10**4,
      2**18: 10**4,
      2**19: 10**3,
      2**20: 10**3,
      2**21: 10**3,
      2**22: 10**3,
      2**23: 10**2,
      2**24: 10**2,
      2**25: 10**1,
      2**26: 10**1,
      2**27: 10**1,
      2**28: 10**1,
    },
    'src2': {
      2**8 : 10**5,
      2**9 : 10**5,
      2**10: 10**5,
      2**11: 10**5,
      2**12: 10**5,
      2**13: 10**4,
      2**14: 10**4,
      2**15: 10**4,
      2**16: 10**4,
      2**17: 10**3,
      2**18: 10**3,
      2**19: 10**3,
      2**20: 10**3,
      2**21: 10**2,
      2**22: 10**2,
      2**23: 10**2,
      2**24: 10**1,
      2**25: 10**1,
      2**26: 10**1,
      2**27: 10**1,
      2**28: 10**1,
    }
  }
  for n, src in [("src1", src1), ("src2", src2)]:
    labels = {
      'CPU': {'data': [], 'names': None},
      'GPU': {'data': [], 'names': None},
    }
    for gsize_pow in range(MIN_GISZE_POW, MAX_GSIZE_POW+1):
      print("##########", gsize_pow, 2**gsize_pow)
      label = opencl.CLDriveLabel(src, num_runs = N_RUNS[n][2**gsize_pow], gsize = 2**gsize_pow, lsize = 256)
      if label != 'ERR':
        labels[label]['data'].append([gsize_pow, 1])
    plt.GroupScatterPlot(
      groups = labels,
      plot_name = "label_per_gsize",
      path = pathlib.Path("./plots/label_gsize/{}".format(n)),
      x_name = "power of 2",
    )
  return

def ExecutionTimesGlobalSize() -> None:
  """
  Iterate over multiple global sizes and collect the execution time
  for transferring to CPU and GPU and executing kernel on CPU and GPU
  and report groupped bar plot.
  """
  global src1
  global src2

  MIN_GISZE_POW = 8
  MAX_GSIZE_POW = 28
  N_RUNS = {
    'src1': {
      2**8 : 10**5,
      2**9 : 10**5,
      2**10: 10**5,
      2**11: 10**5,
      2**12: 10**5,
      2**13: 10**5,
      2**14: 10**4,
      2**15: 10**4,
      2**16: 10**4,
      2**17: 10**4,
      2**18: 10**4,
      2**19: 10**3,
      2**20: 10**3,
      2**21: 10**3,
      2**22: 10**3,
      2**23: 10**2,
      2**24: 10**2,
      2**25: 10**1,
      2**26: 10**1,
      2**27: 10**1,
      2**28: 10**1,
    },
    'src2': {
      2**8 : 10**5,
      2**9 : 10**5,
      2**10: 10**5,
      2**11: 10**5,
      2**12: 10**5,
      2**13: 10**4,
      2**14: 10**4,
      2**15: 10**4,
      2**16: 10**4,
      2**17: 10**3,
      2**18: 10**3,
      2**19: 10**3,
      2**20: 10**3,
      2**21: 10**2,
      2**22: 10**2,
      2**23: 10**2,
      2**24: 10**1,
      2**25: 10**1,
      2**26: 10**1,
      2**27: 10**1,
      2**28: 10**1,
    }
  }
  for n, src in [("src1", src1), ("src2", src2)]:
    labels = {
      'CPU': {'data': [], 'names': None},
      'GPU': {'data': [], 'names': None},
    }
    groups = {
      'cpu_transfer' : [[], []],
      'cpu_kernel'   : [[], []],
      'gpu_transfer' : [[], []],
      'gpu_kernel'   : [[], []],
    }
    for gsize_pow in range(MIN_GISZE_POW, MAX_GSIZE_POW+1):
      print("##########", gsize_pow, 2**gsize_pow)

      cpt, cpk, gpt, gpk = opencl.CLDriveExecutionTimes(src, num_runs = N_RUNS[n][2**gsize_pow], gsize = 2**gsize_pow, lsize = 256)

      if cpt is None:
        while cpt is None:
          cpt, cpk, gpt, gpk = opencl.CLDriveExecutionTimes(src, num_runs = N_RUNS[n][2**gsize_pow], gsize = 2**gsize_pow, lsize = 256)

      print(cpt.mean(), cpk.mean(), gpt.mean(), gpk.mean())

      if not math.isnan(cpt.mean()):
        groups['cpu_transfer'][0].append(lsize_pow)
        groups['cpu_transfer'][1].append(cpt.mean() / (10**6))

      if not math.isnan(cpk.mean()):
        groups['cpu_kernel'][0].append(lsize_pow)
        groups['cpu_kernel'][1].append(cpk.mean() / (10**6))

      if not math.isnan(gpt.mean()):
        groups['gpu_transfer'][0].append(lsize_pow)
        groups['gpu_transfer'][1].append(gpt.mean() / (10**6))

      if not math.isnan(gpk.mean()):
        groups['gpu_kernel'][0].append(lsize_pow)
        groups['gpu_kernel'][1].append(gpk.mean() / (10**6))

    plt.GrouppedBars(
      groups = groups,
      plot_name = "exec_times_per_gsize",
      path = pathlib.Path("./plots/exec_times_gsize/{}".format(n)),
      x_name = "power of 2",
      y_name = "ms",
    )
  return

def ExecutionTimesLocalSize() -> None:
  """
  Iterate over multiple global sizes and collect the execution time
  for transferring to CPU and GPU and executing kernel on CPU and GPU
  and report groupped bar plot.
  """
  global src1
  global src2

  MIN_LISZE_POW = 0
  MAX_LSIZE_POW = 21
  GSIZE_POW = 21
  N_RUNS = 10**2

  for n, src in [("src1", src1), ("src2", src2)]:
    labels = {
      'CPU': {'data': [], 'names': None},
      'GPU': {'data': [], 'names': None},
    }
    groups = {
      'cpu_transfer' : [[], []],
      'cpu_kernel'   : [[], []],
      'gpu_transfer' : [[], []],
      'gpu_kernel'   : [[], []],
    }
    for lsize_pow in range(MIN_LISZE_POW, MAX_LSIZE_POW+1):
      print("##########", lsize_pow, 2**lsize_pow)

      cpt, cpk, gpt, gpk = opencl.CLDriveExecutionTimes(src, num_runs = N_RUNS, gsize = 2**GSIZE_POW, lsize = 2**lsize_pow)

      if cpt is None:
        while cpt is None:
          cpt, cpk, gpt, gpk = opencl.CLDriveExecutionTimes(src, num_runs = N_RUNS, gsize = 2**GSIZE_POW, lsize = 2**lsize_pow)

      print(cpt.mean(), cpk.mean(), gpt.mean(), gpk.mean())

      if not math.isnan(cpt.mean()):
        groups['cpu_transfer'][0].append(lsize_pow)
        groups['cpu_transfer'][1].append(cpt.mean() / (10**6))

      if not math.isnan(cpk.mean()):
        groups['cpu_kernel'][0].append(lsize_pow)
        groups['cpu_kernel'][1].append(cpk.mean() / (10**6))

      if not math.isnan(gpt.mean()):
        groups['gpu_transfer'][0].append(lsize_pow)
        groups['gpu_transfer'][1].append(gpt.mean() / (10**6))

      if not math.isnan(gpk.mean()):
        groups['gpu_kernel'][0].append(lsize_pow)
        groups['gpu_kernel'][1].append(gpk.mean() / (10**6))

    plt.GrouppedBars(
      groups = groups,
      plot_name = "exec_times_per_lsize",
      path = pathlib.Path("./plots/exec_times_lsize/{}".format(n)),
      x_name = "power of 2",
      y_name = "ms",
    )
  return
