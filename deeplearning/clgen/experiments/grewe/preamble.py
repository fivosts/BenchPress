# preamble
import typing
import pathlib
import tempfile
import re
import sys
from collections import Counter

# code for the paper:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import RandomState
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from deeplearning.clgen.experiments.grewe import model
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import plotter

def Finalize(
  output: typing.Optional[typing.Union[str, pathlib.Path]] = None,
  figsize=None,
  tight=True,
  **savefig_opts,
):
  """Finalise a plot.
  Display or show the plot, then close it.
  Args:
    output: Path to save figure to. If not given, plot is shown.
    figsize: Figure size in inches.
    **savefig_opts: Any additional arguments to pass to
      plt.savefig(). Only required if output is not None.
  """
  # Set figure size.
  if figsize is not None:
    plt.gcf().set_size_inches(*figsize)

  # Set plot layout.
  if tight:
    plt.tight_layout()

  if output is None:
    plt.show()
  else:
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), **savefig_opts)
  plt.close()

def mean(array):
  """
  Return the mean value of a list of divisible numbers.
  """
  n = len(array)

  if n < 1:
    return 0
  elif n == 1:
    return array[0]
  return sum(array) / n

# plotting config:
sns.set(style="ticks", color_codes=True)
plt.style.use(["seaborn-white", "seaborn-paper"])

# tables config:
pd.set_option('display.max_rows', 15)

def line_word_char_count(path):
  """count words, lines, chars in file"""
  num_lines = 0
  num_words = 0
  num_chars = 0

  with open(path) as infile:
    for line in infile:
      words = line.split()

      num_lines += 1
      num_words += len(words)
      num_chars += len(line)

  return num_lines, num_words, num_chars


def rand_jitter(arr, factor=0.01, randomstate=RandomState(204)):
  """apply jitter to array"""
  stdev = factor * (max(arr) - min(arr))
  return arr + randomstate.randn(len(arr)) * stdev


def scatter_with_jitter(plt, x, y, **kwargs):
  """scatter x,y values with jitter"""
  jitter_opts = kwargs.get("jitter_opts", {})
  if "jitter_opts" in kwargs:
    kwargs.pop("jitter_opts")

  return plt.scatter(rand_jitter(x, **jitter_opts),
                     rand_jitter(y, **jitter_opts), **kwargs)


def shortlabels(groups):
  """shorten benchmark suite names"""
  return [escape_suite_name(re.sub("-.+$", "", x)) for x in groups]


def shortbenchmark(benchmark):
  """short benchmark name"""
  return benchmark.split('-')[-1]


def escape_benchmark_name(g):
  """escape benchmark name for display"""
  c = g.split('-')
  return escape_suite_name(g) + "." + c[-2]


def plot_pca(X, B_out, Bother=None, pca=None):
  """plot PCA projection of feature space"""

  def jitter_opts(randomstate):
    return {"factor": .075, "randomstate": RandomState(randomstate)}

  # size and opacity
  plot_opts = {"s": 85, "alpha": .65}

  # apply jitter and repack
  x, y = zip(*X)
  x = rand_jitter(x, **jitter_opts(204))
  y = rand_jitter(y, **jitter_opts(205))
  X = list(zip(x, y))

  # group by correct or not
  correct = [x for x, b in zip(X, B_out.to_dict('records')) if b["p_correct"]]
  incorrect = [x for x, b in zip(X, B_out.to_dict('records')) if
               not b["p_correct"]]

  if Bother is not None:
    additional = pca.transform(get_raw_features(Bother))
    scatter_with_jitter(plt, *zip(*additional), color="g", marker="o",
                        label="Additional", jitter_opts=jitter_opts(206),
                        **plot_opts)
  plt.scatter(*zip(*incorrect),
              color="r", marker="v", label='Incorrect', **plot_opts)
  plt.scatter(*zip(*correct),
              color="b", marker="^", label='Correct', **plot_opts)

  # no tick labels
  ax = plt.gca()
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  # axis labels
  plt.xlabel(r"Principle Component 1 $\rightarrow$", ha="right")
  plt.ylabel(r"Principle Component 2 $\rightarrow$", ha="right")

  # position axis labels at end of axis
  ax.xaxis.set_label_coords(1, -.025)
  ax.yaxis.set_label_coords(-.025, 1)

  # show legend
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles[::-1], labels[::-1])
  ax.get_legend().draw_frame(True)

  return ax


def get_our_model():
  """return extended model"""
  return KNeighborsClassifier(1)


def get_our_features(D):
  """return extended featureset"""
  return np.array([
    D["comp"].values,
    D["rational"].values,
    D["mem"].values,
    D["localmem"].values,
    D["coalesced"].values,
    D["transfer"].values,
    D["wgsize"].values,
    (D["transfer"].values / (D["comp"].values + D["mem"].values)),
    (D["coalesced"].values / D["mem"].values),
    ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),
    (D["comp"].values / D["mem"].values),
  ]).T


def get_raw_features(D):
  """return raw feature values"""
  return np.array([
    D["comp"].values,
    D["rational"].values,
    D["mem"].values,
    D["localmem"].values,
    D["coalesced"].values,
    D["atomic"].values,
    D["transfer"].values,
    D["wgsize"].values,
  ]).T


def get_cgo13_features(D):
  """return features used in CGO'13"""
  return np.array([
    (D["transfer"].values / (D["comp"].values + D["mem"].values)),
    (D["coalesced"].values / D["mem"].values),
    ((D["localmem"].values / D["mem"].values) * D["wgsize"].values),
    (D["comp"].values / D["mem"].values),
  ]).T


def readfile(path):
  """read file to string"""
  with open(path) as infile:
    return ''.join(infile.readlines())


def escape_suite_name(g):
  """format benchmark suite name for display"""
  c = g.split('-')
  if (c[0] == "amd" or c[0] == "npb" or c[0] == "nvidia" or c[0] == "shoc"):
    return c[0].upper()
  else:
    return c[0].capitalize()


def get_nearest_neighbour_distance(F1, F2):
  """return nearest-neighbour distances from F1 to F2"""
  nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(F2)
  distances, indices = nbrs.kneighbors(F1)
  return distances

def plot_speedups_with_clgen(benchmarks_data, clgen_data, synth_bench_name = "CLgen", suite=""):
  """
  Plot speedups of predictive models trained with and without clgen.

  Returns speedups (without and with).
  """
  # datasets: B - benchmarks, S - synthetics, BS - benchmarks + synthetics:
  B = pd.read_csv(benchmarks_data)
  B["group"] = ["B"] * len(B)

  S = pd.read_csv(clgen_data)
  S["group"] = ["S"] * len(S)

  BS = pd.concat((B, S))

  # find the ZeroR. This is the device which is most frequently optimal
  Bmask = B[B["benchmark"].str.contains(suite)]
  zeror = Counter(Bmask["oracle"]).most_common(1)[0][0]
  zeror_runtime = "runtime_" + zeror.lower()

  # get the names of the benchmarks, in the form: $suite-$version-$benchmark
  benchmark_names = sorted(set([
    re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)-", b).group(1)
    for b in B["benchmark"] if b.startswith(suite)
  ]))

  B_out, S_out, BS_out = [], [], []
  for benchmark in benchmark_names:
    clf = model.model()
    features = get_cgo13_features
    # cross validate on baseline
    B_out += model.leave_one_benchmark_out(clf, features, B, benchmark)
    # reset model
    clf = model.model()
    S_out += model.leave_one_benchmark_out(clf, features, BS, benchmark, synthetics = True, is_clgen = True)
    clf = model.model()
    # repeate cross-validation with synthetic kernels
    BS_out += model.leave_one_benchmark_out(clf, features, BS, benchmark, synthetics = False, is_clgen = True)

  # create results frame
  R_out = []
  for b, s, bs in zip(B_out, S_out, BS_out):
    # get runtimes of device using predicted device
    b_p_runtime = b["runtime_" + b["p"].lower()]
    s_p_runtime = s["runtime_" + s["p"].lower()]
    bs_p_runtime = bs["runtime_" + bs["p"].lower()]
    best_possible_p = b["runtime_" + b["oracle"].lower()]

    # speedup is the ratio of runtime using the predicted device
    # over runtime using ZeroR device
    b["p_speedup"] =  b[zeror_runtime] / b_p_runtime
    s["p_speedup"] = s[zeror_runtime] / s_p_runtime
    bs["p_speedup"] = bs[zeror_runtime] / bs_p_runtime

    b["best_p_speedup"] = b[zeror_runtime] / best_possible_p

    b["opt_runtime"] = b_p_runtime
    s["opt_runtime"] = s_p_runtime
    bs["opt_runtime"] = bs_p_runtime

    # print(b_p_runtime, s_p_runtime, bs_p_runtime, b[zeror_runtime], s[zeror_runtime], bs[zeror_runtime])

    if "training" in benchmarks_data:
      # $benchmark
      group = escape_benchmark_name(b["benchmark"])
    else:
      # $benchmark.$dataset
      group = re.sub(r"[^-]+-[0-9\.]+-([^-]+)-.+", r"\1",
                     b["benchmark"]) + "." + str(b["dataset"])
    b["group"] = group
    s["group"] = group
    bs["group"] = group

    # set the training data type
    b["training"] = "Grewe et al."
    s["training"] = "Only {}".format(synth_bench_name)
    bs["training"] = "w. {}".format(synth_bench_name)

    R_out.append(b)
    R_out.append(s)
    R_out.append(bs)

  R = pd.DataFrame(R_out)

  b_mask = R["training"] == "Grewe et al."
  s_mask = R["training"] == "Only {}".format(synth_bench_name)
  bs_mask = R["training"] == "w. {}".format(synth_bench_name)

  b_gpu = (len(R[b_mask][R[b_mask]["oracle"] == "GPU"]), len(R[b_mask][R[b_mask]["oracle"] == "GPU"][R[b_mask]["p"] == "GPU"]), len(R[b_mask][R[b_mask]["oracle"] == "GPU"][R[b_mask]["p"] == "CPU"]))
  b_cpu = (len(R[b_mask][R[b_mask]["oracle"] == "CPU"]), len(R[b_mask][R[b_mask]["oracle"] == "CPU"][R[b_mask]["p"] == "GPU"]), len(R[b_mask][R[b_mask]["oracle"] == "CPU"][R[b_mask]["p"] == "CPU"]))

  s_gpu = (len(R[s_mask][R[s_mask]["oracle"] == "GPU"]), len(R[s_mask][R[s_mask]["oracle"] == "GPU"][R[s_mask]["p"] == "GPU"]), len(R[s_mask][R[s_mask]["oracle"] == "GPU"][R[s_mask]["p"] == "CPU"]))
  s_cpu = (len(R[s_mask][R[s_mask]["oracle"] == "CPU"]), len(R[s_mask][R[s_mask]["oracle"] == "CPU"][R[s_mask]["p"] == "GPU"]), len(R[s_mask][R[s_mask]["oracle"] == "CPU"][R[s_mask]["p"] == "CPU"]))

  bs_gpu = (len(R[bs_mask][R[bs_mask]["oracle"] == "GPU"]), len(R[bs_mask][R[bs_mask]["oracle"] == "GPU"][R[bs_mask]["p"] == "GPU"]), len(R[bs_mask][R[bs_mask]["oracle"] == "GPU"][R[bs_mask]["p"] == "CPU"]))
  bs_cpu = (len(R[bs_mask][R[bs_mask]["oracle"] == "CPU"]), len(R[bs_mask][R[bs_mask]["oracle"] == "CPU"][R[bs_mask]["p"] == "GPU"]), len(R[bs_mask][R[bs_mask]["oracle"] == "CPU"][R[bs_mask]["p"] == "CPU"]))

  print("{} GPU Oracle Grewe: {} GPU / {} CPU".format(b_gpu[0], b_gpu[1], b_gpu[2]))
  print("{} GPU Oracle Only {}: {} GPU / {} CPU".format(s_gpu[0], synth_bench_name, s_gpu[1], s_gpu[2]))
  print("{} GPU Oracle Grewe + {}: {} GPU / {} CPU".format(bs_gpu[0], synth_bench_name, bs_gpu[1], bs_gpu[2]))
  print()
  print("{} CPU Oracle Grewe: {} GPU / {} CPU".format(b_cpu[0], b_cpu[1], b_cpu[2]))
  print("{} CPU Oracle Only {}: {} GPU / {} CPU".format(s_cpu[0], synth_bench_name, s_cpu[1], s_cpu[2]))
  print("{} CPU Oracle Grewe + {}: {} GPU / {} CPU".format(bs_cpu[0], synth_bench_name, bs_cpu[1], bs_cpu[2]))

  B_speedup = mean(R[b_mask].groupby(["group"])["p_speedup"].mean())
  S_speedup = mean(R[s_mask].groupby(["group"])["p_speedup"].mean())
  BS_speedup = mean(R[bs_mask].groupby(["group"])["p_speedup"].mean())

  B_runtimes = R[b_mask]["opt_runtime"].mean()
  S_runtimes = R[s_mask]["opt_runtime"].mean()
  BS_runtimes = R[bs_mask]["opt_runtime"].mean()

  groups = {
    "Benchmarks": {},
    "Bench+Synth": {},
    "Synthetics": {},
  }

  bench_times = 0.0
  benchsynth_times = 0.0
  synth_times = 0.0

  R.to_csv("./investigate.csv")
  # for x in R[b_mask]:
  #   print(x)
  #   bench_times += x["runtime_" + x["p"].lower()]

  # for x in R[bs_mask]:
  #   benchsynth_times += x["runtime_" + x["p"].lower()]

  # for x in R[s_mask]:
  #   synth_times += x["runtime_" + x["p"].lower()]

  # print(bench_times)
  # print(benchsynth_times)
  # print(synth_times)

  print("Only bench: {}".format(B_runtimes))
  print("Only ML: {}".format(S_runtimes))
  print("Bench + ML: {}".format(BS_runtimes))

  for x in R[b_mask]["p_speedup"]:
    x = int(x)
    if x not in groups["Benchmarks"]:
      groups["Benchmarks"][x] = 1
    else:
      groups["Benchmarks"][x] += 1

  for x in R[bs_mask]["p_speedup"]:
    x = int(x)
    if x not in groups["Bench+Synth"]:
      groups["Bench+Synth"][x] = 1
    else:
      groups["Bench+Synth"][x] += 1

  for x in R[s_mask]["p_speedup"]:
    x = int(x)
    if x not in groups["Synthetics"]:
      groups["Synthetics"][x] = 1
    else:
      groups["Synthetics"][x] += 1

  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plotter.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "speedup_distribution",
    path = pathlib.Path("."),
    title = "Speedup distribution frequency",
    x_name = "Speedup absolute value",
  )

  b_distr = distributions.GenericDistribution([int(x) for x in R[b_mask]["p_speedup"]], "plots", "benchmarks")
  s_distr = distributions.GenericDistribution([int(x) for x in R[s_mask]["p_speedup"]], "plots", "synthetics")
  bs_distr = distributions.GenericDistribution([int(x) for x in R[bs_mask]["p_speedup"]], "plots", "synthetics_benchmarks")

  b_distr.plot()
  s_distr.plot()
  bs_distr.plot()

  print(s_distr - b_distr > 0)
  print(bs_distr - b_distr > 0)

  (s_distr - b_distr).plot()
  (bs_distr - b_distr).plot()

  print("  #. benchmarks:                  ",
        len(set(B["benchmark"])), "kernels,", len(B), "observations")
  print("  #. synthetic:                   ",
        len(set(S["benchmark"])), "kernels,", len(S), "observations")
  print()
  print("  ZeroR device:                    {}".format(zeror))
  print()
  print("  Speedup of Grewe et al.:         {:.2f} x".format(B_speedup))
  print("  Speedup w. {}:                {:.2f} x".format(synth_bench_name, BS_speedup))
  print("  Speedup Only {}:              {:.2f} x".format(synth_bench_name, S_speedup))
  print("  Geo Speedup of Grewe et al.:         {:.2f} x".format(model.geomean([x for x in R[b_mask]["p_speedup"]])))
  print("  Geo Speedup w. {}:                {:.2f} x".format(synth_bench_name, model.geomean([x for x in R[bs_mask]["p_speedup"]])))
  print("  Geo Speedup Only {}:              {:.2f} x".format(synth_bench_name, model.geomean([x for x in R[s_mask]["p_speedup"]])))
  print("  Best speedup {}:              {:.2f} x".format("Best", model.geomean([x for x in R[b_mask]["best_p_speedup"]])))

  # bft = [x.p_speedup for idx, x in R[b_mask].iterrows() if x.group == "FT.B"]
  # sft = [x.p_speedup for idx, x in R[s_mask].iterrows() if x.group == "FT.B"]
  # bsft = [x.p_speedup for idx, x in R[bs_mask].iterrows() if x.group == "FT.B"]

  print()
  print()
  print()

  # print("FT.B Grewe: {}".format(sum(bft) / len(bft)))
  # print("FT.B w {}: {}".format(synth_bench_name, sum(bsft) / len(bsft)))
  # print("FT.B Only {}: {}".format(synth_bench_name, sum(sft) / len(sft)))

  print("Total execution times:")
  print()

  R = R.append({  # average bars
    "group": "Average",
    "p_speedup": B_speedup,
    "training": "Grewe et al."
  }, ignore_index=True)
  R = R.append({
    "group": "Average",
    "p_speedup": BS_speedup,
    "training": "w. {}".format(synth_bench_name)
  }, ignore_index=True)

  R["p_speedup"] -= 1  # negative offset so that bars start at 1

  # colors
  palette = sns.cubehelix_palette(len(set(R["training"])),
                                  rot=-.4, light=.85, dark=.35)

  ax = sns.barplot(
      x="group", y="p_speedup", data=R, ci=None, hue="training",
      palette=palette)
  plt.ylabel("Speedup")
  plt.xlabel("")

  plt.axhline(y=0, color="k", lw=1)  # speedup line
  plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1,
              linestyle="--")  # average line

  ax.get_legend().set_title("")  # no legend title
  plt.legend(loc='upper right')
  ax.get_legend().draw_frame(True)

  # plot shape and size
  figsize = (3*9, 3*2.2)
  if "nvidia" in benchmarks_data:
    typecast = int;
    plt.ylim(-1, 16)
  elif "training" in benchmarks_data:
    typecast = float;
    figsize = (3*7, 3*3.2)
  else:
    typecast = float

  # counter negative offset:
  ax.set_yticklabels([typecast(i) + 1 for i in ax.get_yticks()])

  plt.setp(ax.get_xticklabels(), rotation=90)

  Finalize(output = "plot.png", figsize=figsize, tight=True)
  return B_speedup, BS_speedup


def _compare_clfs(clf1, get_features1, clf2, get_features2, D1, D2, benchmark):
  """cross-validate across all benchmarks using CGO13 model and our own, with
  and without synthetic benchmarks. Report per-platform speedup of our model
  over CGO13"""
  test1_mask = D1["benchmark"].str.contains(r"^" + benchmark)
  test2_mask = D2["benchmark"].str.contains(r"^" + benchmark)
  assert (len(D1[test1_mask]) == len(D2[test2_mask]))

  # create data masks. For training we exclude all results from benchmark
  train1_mask = ~test1_mask
  train2_mask = ~test2_mask

  # create training and testing data
  X1_train = get_features1(D1.loc[train1_mask])
  X2_train = get_features2(D2.loc[train2_mask])
  y1_train = model.getlabels(D1[train1_mask])
  y2_train = model.getlabels(D2[train2_mask])

  D1_test = D1[test1_mask]
  D2_test = D2[test2_mask]
  X1_test = get_features1(D1.loc[test1_mask])
  X2_test = get_features2(D2.loc[test2_mask])
  y1_test = model.getlabels(D1_test)
  y2_test = model.getlabels(D2_test)

  clf1.fit(X1_train, y1_train)  # train classifiers
  clf2.fit(X2_train, y2_train)

  predicted1 = clf1.predict(X1_test)  # make predictions
  predicted2 = clf2.predict(X2_test)

  D_out = []
  for d, y, p1, p2 in zip(D1_test.to_dict('records'), y1_test,
                          predicted1, predicted2):
    d["p1"], d["p2"] = p1, p2
    D_out.append(d)

  return D_out  # return a list of dicts


def plot_speedups_extended_model_2platform(platform_a, platform_b):
  """
  Plot speedup of extended model over Grewe et al for 2 platforms
  """
  aB = pd.read_csv(platform_a[0])
  aB["synthetic"] = np.zeros(len(aB))
  bB = pd.read_csv(platform_b[0])
  bB["synthetic"] = np.zeros(len(bB))
  B = pd.concat((aB, bB))

  aS = pd.read_csv(platform_a[1])
  aS["synthetic"] = np.ones(len(aS))
  bS = pd.read_csv(platform_b[1])
  bS["synthetic"] = np.ones(len(bS))
  S = pd.concat((aS, bS))

  aBS = pd.concat((aB, aS))
  bBS = pd.concat((bB, bS))
  BS = pd.concat((B, S))

  assert (len(B) == len(aB) + len(bB))  # sanity checks
  assert (len(S) == len(aS) + len(bS))
  assert (len(BS) == len(aBS) + len(bBS))

  # get benchmark names: <suite>-<benchmark>
  benchmark_names = sorted(set([
    re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
    for b in B["benchmark"]
  ]))

  # perform cross-validation
  B_out = []
  for i, benchmark in enumerate(benchmark_names):
    print("\ranalyzing", i + 1, benchmark, end="")
    cgo13_clf, our_clf = model.model(), get_our_model()
    cgo13_features, our_features = get_cgo13_features, get_our_features

    # cross validate on Grewe et al. and our model
    tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                        aBS, aBS, benchmark)
    for d in tmp: d["platform"] = "AMD Tahiti 7970"
    B_out += tmp

    # reset models
    cgo13_clf, our_clf = model.model(), get_our_model()

    # same as before, on other platform:
    tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                        bBS, bBS, benchmark)
    for d in tmp: d["platform"] = "NVIDIA GTX 970"
    B_out += tmp
  print()

  # create results frame
  R_out = []
  # get runtimes of device using predicted device
  for b in B_out:
    p1_runtime = b["runtime_" + b["p1"].lower()]
    p2_runtime = b["runtime_" + b["p2"].lower()]

    # speedup is the ratio of runtime using our predicted device
    # over runtime using CGO13 predicted device.
    b["p_speedup"] = p2_runtime / p1_runtime

    # get the benchmark name
    b["group"] = escape_benchmark_name(b["benchmark"])

    R_out.append(b)
  R = pd.DataFrame(R_out)

  improved = R[R["p_speedup"] > 1]

  Amask = R["platform"] == "AMD Tahiti 7970"
  Bmask = R["platform"] == "NVIDIA GTX 970"
  a = R[Amask]
  b = R[Bmask]

  a_speedups = a.groupby(["group"])["p_speedup"].mean()
  b_speedups = b.groupby(["group"])["p_speedup"].mean()

  a_speedup = mean(a_speedups)
  b_speedup = mean(b_speedups)

  assert (len(R) == len(a) + len(b))  # sanity-check

  print("  #. benchmarks:          ",
        len(set(B["benchmark"])), "kernels,", len(B), "observations")
  print("  #. synthetic:           ",
        len(set(S["benchmark"])), "kernels,", len(S), "observations")
  print()
  print("  Speedup on AMD:          {:.2f} x".format(a_speedup))
  print("  Speedup on NVIDIA:       {:.2f} x".format(b_speedup))

  palette = sns.cubehelix_palette(
      len(set(R["platform"])), start=4, rot=.8, light=.8, dark=.3)

  R = R.append({  # average bars
    "group": "Average",
    "p_speedup": a_speedup,
    "platform": "AMD Tahiti 7970"
  }, ignore_index=True)
  R = R.append({
    "group": "Average",
    "p_speedup": b_speedup,
    "platform": "NVIDIA GTX 970"
  }, ignore_index=True)

  R["p_speedup"] -= 1  # negative offset so that bars start at 1

  ax = sns.barplot(x="group", y="p_speedup", hue="platform", data=R,
                   palette=palette, ci=None)

  plt.ylabel("Speedup over Grewe et al.");
  plt.xlabel("")

  plt.axhline(y=0, color="k", lw=1)
  plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1, linestyle="--")
  plt.ylim(-1, 9)
  plt.setp(ax.get_xticklabels(), rotation=90)  # rotate x ticks
  ax.get_legend().set_title("")  # legend
  plt.legend(loc='upper right')

  # counter negative offset
  ax.set_yticklabels([int(i) + 1 for i in ax.get_yticks()])

  ax.get_legend().draw_frame(True)

  Finalize(figsize=(9, 4), tight=True)


def plot_speedups_extended_model(benchmarks_data, clgen_data):
  """
  Plots speedups of extended model over Grewe et al

  Returns: speedup
  """
  B = pd.read_csv(benchmarks_data)
  B["synthetic"] = np.zeros(len(B))

  S = pd.read_csv(clgen_data)
  S["synthetic"] = np.ones(len(S))

  BS = pd.concat((B, S))

  assert (len(BS) == len(B) + len(S))

  # get benchmark names: <suite>-<benchmark>
  benchmark_names = sorted(set([
    re.match(r"^([^0-9]+-[0-9\.]+-[^-]+)", b).group(1)
    for b in B["benchmark"]
  ]))

  # perform cross-validation
  B_out = []
  for i, benchmark in enumerate(benchmark_names):
    print("\ranalyzing", i + 1, benchmark, end="")
    cgo13_clf, our_clf = model.model(), get_our_model()
    cgo13_features, our_features = get_cgo13_features, get_our_features

    # cross validate on Grewe et al. and our model
    tmp = _compare_clfs(cgo13_clf, cgo13_features, our_clf, our_features,
                        BS, BS, benchmark)
    B_out += tmp
  print()

  # create results frame
  R_out = []
  # get runtimes of device using predicted device
  for b in B_out:
    p1_runtime = b["runtime_" + b["p1"].lower()]
    p2_runtime = b["runtime_" + b["p2"].lower()]

    # speedup is the ratio of runtime using our predicted device
    # over runtime using CGO13 predicted device.
    b["p_speedup"] = p2_runtime / p1_runtime

    # get the benchmark name
    b["group"] = escape_benchmark_name(b["benchmark"])

    R_out.append(b)
  R = pd.DataFrame(R_out)

  improved = R[R["p_speedup"] > 1]

  speedups = R.groupby(["group"])["p_speedup"].mean()
  speedup = mean(speedups)

  print("  #. benchmarks:          ",
        len(set(B["benchmark"])), "kernels,", len(B), "observations")
  print("  #. synthetic:           ",
        len(set(S["benchmark"])), "kernels,", len(S), "observations")
  print()
  print("  Speedup:                 {:.2f} x".format(speedup))

  palette = sns.cubehelix_palette(1, start=4, rot=.8, light=.8, dark=.3)

  R = R.append({  # average bar
    "group": "Average",
    "p_speedup": speedup
  }, ignore_index=True)

  R["p_speedup"] -= 1  # negative offset so that bars start at 1

  ax = sns.barplot(x="group", y="p_speedup", data=R,
                   palette=palette, ci=None)

  plt.ylabel("Speedup over Grewe et al.");
  plt.xlabel("")

  plt.axhline(y=0, color="k", lw=1)
  plt.axvline(x=plt.xlim()[1] - 1, color="k", lw=1, linestyle="--")
  plt.ylim(-1, 9)
  plt.setp(ax.get_xticklabels(), rotation=90)  # rotate x ticks

  # counter negative offset
  ax.set_yticklabels([int(i) + 1 for i in ax.get_yticks()])

  Finalize(figsize=(7, 3.7), tight=True)
  return speedup

if __name__ == "__main__":

  base_path = pathlib.Path("/private/home/foivos/pact_grewe_csv_edited")

  baselines = [
    base_path / "baseline" / "gpgpu_benchmarks_cc1.csv",
    # base_path / "baseline" / "gpgpu_cc3.csv",
    # base_path / "baseline" / "nvidia-benchmarks.csv",
  ]

  me = base_path / "me" / "BenchPress_AL.csv"
  passive = base_path / "me" / "BenchPress_passive.csv"
  clgen = base_path / "others" / "CLgen.csv"

  github = base_path / "others" / "GitHub.csv"
  github_extended = base_path / "others" / "GitHub-extended.csv"

  for baseline in baselines:
    # plot_speedups_with_clgen(
    #   open(baseline, 'r'),
    #   open(me, 'r'),
    #   synth_bench_name = "BenchPress",
    # )
    # print()
    # print()
    # print()
    plot_speedups_with_clgen(
      open(baseline, 'r'),
      open(passive, 'r'),
      synth_bench_name = "GitHub",
    )
    plot_speedups_with_clgen(
      open(baseline, 'r'),
      open(clgen, 'r'),
      synth_bench_name = "CLgen",
    )
    # print()
    # print()
    # print()
    # plot_speedups_with_clgen(
    #   open(baseline, 'r'),
    #   open(github_extended, 'r'),
    #   synth_bench_name = "GitHub_extended",
    # )
    # input()

#   plot_speedups_with_clgen(
#     open("/var/foivos/results/clgen_paper_artifacts/nvidia-benchmarks.csv", 'r'),
#     open("/var/foivos/results/clgen_paper_artifacts/nvidia-clgen.csv", 'r')
#   )

#   db_path = pathlib.Path("/var/foivos/results/pldi_results/BERT/Fixed_input/samples.db")
#   db = samples_database.SamplesDatabase(db_path, must_exist = True)
#   d = api.ToDataFrame(db)
#   d.to_csv(str(db_path.parent / "samples_dataframe.csv"))
#   with tempfile.NamedTemporaryFile("w", prefix="preamble_", suffix=".csv") as f:
#     plot_speedups_with_clgen(
#       open("/var/foivos/results/clgen_paper_artifacts/nvidia-benchmarks.csv", 'r'),
#       open("/var/foivos/results/pldi_results/BERT/Fixed_input/samples_dataframe.csv", 'r'),
#     )
#   exit()
