from deeplearning.benchpress.util import plotter as plt
import pathlib

groups = {
	"BenchDirect": {},
	"BenchPress": {},
}

## Grewe
groups["BenchDirect"]['data'] = [[267*2048, 73.56], [266*1024, 77.79], [512*290, 81.56], [256*289, 82.94], [128*272, 85.30], [64*282, 87.62], [32*151, 96.24]]
groups["BenchPress"]['data'] = [[2048*286, 76.79], [1024*306, 83.62], [512*325, 88.27], [256*326, 91.47], [128*333, 95.53], [64*338, 97.30], [32*236, 99.13]]

# relative proximity
groups["BenchDirect"]['data'] = [[267*2048, 80.99], [266*1024, 77.07], [512*290, 72.45], [256*289, 68.75], [128*272, 61.65], [64*282, 56.97], [32*151, 45.06]]
groups["BenchPress"]['data'] = [[2048*286, 75.83], [1024*306, 69.44], [512*325, 62.23], [256*326, 55.68], [128*333, 48.27], [64*338, 42.16], [32*236, 34.67]]

groups["BenchDirect"]['names'] = [2048, 1024, 512, 256, 128, 64, 32]
groups["BenchPress"]['names'] = [2048, 1024, 512, 256, 128, 64, 32]

time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

print(time_speedup)
print(acc_speedup)

plt.GroupScatterPlot(
  groups,
  plot_name="pareto_grewe",
  path = pathlib.Path("./pareto").resolve(),
  title = "Grewe Features",
  x_name = "# Total Inferences",
  y_name = "Avg Relative Proximity (%)",
  showline_x = True,
  showline_y = True,
  linewidth = 2,
  linecolor = "black",
  showgrid_x = False,
  showgrid_y = True,
  bg_color = "white",
  gridcolor_x = "gray",
  gridcolor_y = "gray",
  gridwidth = 1,
  height = 900,
  width = 1280,
  tickfont = 24,
  axisfont = 24,
  legendfont = 18,
  titlefont = 24,
  legend_x = 0.75,
  legend_y = 0.45,
)
plt.GroupScatterPlot(
	{"% Speedup": {'data': time_speedup, 'names': []}, "% Proximity": {'data': acc_speedup, 'names': []}},
	plot_name="speedup_grewe",
	path = pathlib.Path("./pareto").resolve(),
	title = "",
	marker_style = [dict(color = "rgb(57, 105, 172)"), dict(color = "rgb(102, 166, 30)")],
	x_name = "Workload Size",
	y_name = "% Gain over BenchPress",
	showline_x = True,
	showline_y = True,
	linewidth = 2,
	linecolor = "black",
	showgrid_x = False,
	showgrid_y = True,
	bg_color =  "white",
	gridcolor_x = "gray",
	gridcolor_y = "gray",
	gridwidth = 1,
	height = 900,
	width = 1280,
	tickfont = 24,
	axisfont = 24,
	legendfont = 18,
	titlefont = 24,
	legend_x = 0.75,
    legend_y = 0.87,
)

## Autophase
groups["BenchDirect"]['data'] = [[262*2048, 41.02], [262*1024, 44.7], [512*267, 52.36], [256*262, 54.60], [128*254, 58.02], [64*230, 61.09], [32*164, 57.74]]
groups["BenchPress"]['data'] = [[2048*292, 48.88], [1024*297, 50.84], [512*302, 57.38], [256*307, 57.63], [128*312, 71.32], [64*312, 74.27], [32*254, 83.59]]

# relative proximity
groups["BenchDirect"]['data'] = [[267*2048, 74.63], [266*1024, 72.03], [512*290, 66.77], [256*289, 64.39], [128*272, 61.38], [64*282, 59.22], [32*151, 57.81]]
groups["BenchPress"]['data'] = [[2048*286, 64.51], [1024*306, 65.78], [512*325, 60.08], [256*326, 58.19], [128*333, 57.81], [64*338, 43.82], [32*236, 33.32]]


time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

print(time_speedup)
print(acc_speedup)

plt.GroupScatterPlot(
  groups,
  plot_name="pareto_autophase",
  path = pathlib.Path("./pareto").resolve(),
  title = "Autophase Features",
  x_name = "# Total Inferences",
  y_name = "Avg Relative Proximity (%)",
  showline_x = True,
  showline_y = True,
  linewidth = 2,
  linecolor = "black",
  showgrid_x = False,
  showgrid_y = True,
  bg_color = "white",
  gridcolor_x = "gray",
  gridcolor_y = "gray",
  gridwidth = 1,
  height = 900,
  width = 1280,
  tickfont = 24,
  axisfont = 24,
  legendfont = 18,
  titlefont = 24,
  legend_x = 0.75,
  legend_y = 0.52,
)
plt.GroupScatterPlot(
	{"% Speedup": {'data': time_speedup, 'names': []}, "% Proximity": {'data': acc_speedup, 'names': []}},
	plot_name="speedup_autophase",
	path = pathlib.Path("./pareto").resolve(),
	title = "",
	marker_style = [dict(color = "rgb(57, 105, 172)"), dict(color = "rgb(102, 166, 30)")],
	x_name = "Workload Size",
	y_name = "% Gain over BenchPress",
	showline_x = True,
	showline_y = True,
	linewidth = 2,
	linecolor = "black",
	showgrid_x = False,
	showgrid_y = True,
	bg_color =  "white",
	gridcolor_x = "gray",
	gridcolor_y = "gray",
	gridwidth = 1,
	height = 900,
	width = 1280,
	tickfont = 24,
	axisfont = 24,
	legendfont = 18,
	titlefont = 24,
	legend_x = 0.75,
    legend_y = 0.87,
)

## Instcount
groups["BenchDirect"]['data'] = [[252*2048, 30.73], [257*1024, 34.36], [512*262, 36.32], [256*259, 39.89], [128*265, 41.96], [64*257, 46.21], [32*163, 48.33]]
groups["BenchPress"]['data'] = [[2048*301, 32.63], [1024*307, 40.09], [512*302, 40.49], [256*307, 52.89], [128*307, 56.41], [64*312, 57.77], [32*208, 69.11]]

# relative proximity
groups["BenchDirect"]['data'] = [[267*2048, 79.86], [266*1024, 77.90], [512*290, 76.27], [256*289, 73.92], [128*272, 71.10], [64*282, 66.54], [32*151, 63.33]]
groups["BenchPress"]['data'] = [[2048*286, 75.57], [1024*306, 70.61], [512*325, 71.79], [256*326, 59.43], [128*333, 59.24], [64*338, 54.92], [32*236, 41.41]]

time_speedup = [100*abs(round((x[0]-y[0])) / y[0]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]
acc_speedup = [100*abs(round((x[1]-y[1])) / y[1]) for x, y in zip(groups["BenchDirect"]["data"], groups["BenchPress"]["data"])]

time_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], time_speedup)]
acc_speedup = [[x, y] for x, y in zip([2048, 1024, 512, 256, 128, 64, 32], acc_speedup)]

print(time_speedup)
print(acc_speedup)

plt.GroupScatterPlot(
  groups,
  plot_name="pareto_instcount",
  path = pathlib.Path("./pareto").resolve(),
  title = "InstCount Features",
  x_name = "# Total Inferences",
  y_name = "Avg Relative Proximity (%)",
  showline_x = True,
  showline_y = True,
  linewidth = 2,
  linecolor = "black",
  showgrid_x = False,
  showgrid_y = True,
  bg_color = "white",
  gridcolor_x = "gray",
  gridcolor_y = "gray",
  gridwidth = 1,
  height = 900,
  width = 1280,
  tickfont = 24,
  axisfont = 24,
  legendfont = 18,
  titlefont = 24,
  legend_x = 0.75,
  legend_y = 0.48,
)
plt.GroupScatterPlot(
	{"% Speedup": {'data': time_speedup, 'names': []}, "% Proximity": {'data': acc_speedup, 'names': []}},
	plot_name="speedup_instcount",
	path = pathlib.Path("./pareto").resolve(),
	marker_style = [dict(color = "rgb(57, 105, 172)"), dict(color = "rgb(102, 166, 30)")],
	title = "",
	x_name = "Workload Size",
	y_name = "% Gain over BenchPress",
	showline_x = True,
	showline_y = True,
	linewidth = 2,
	linecolor = "black",
	showgrid_x = False,
	showgrid_y = True,
	bg_color =  "white",
	gridcolor_x = "gray",
	gridcolor_y = "gray",
	gridwidth = 1,
	height = 900,
	width = 1280,
	tickfont = 24,
	axisfont = 24,
	legendfont = 18,
	titlefont = 24,
	legend_x = 0.75,
    legend_y = 0.87,
)


