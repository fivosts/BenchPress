import numpy as np
from scipy.stats import norm
from plotly import graph_objs as go

def FrequencyBars(x: np.array,
                  y: np.array,
                  ) -> None:
  """Plot frequency bars based on key."""
  fig = go.Figure()
  fig.add_trace(
    go.Bar(
      x = x,
      y = y,
      showlegend = False,
      marker_color = '#ac3939',
      opacity = 0.75,
    )
  )
  fig.write_html ("./test.html")
  return

# stage0 = [0, 1, 2, 3, 4, 5]
# stage1 = stage0 + [6, 7, 8, 9, 10] *2
# stage2 = stage1 + [11, 12, 13,  14, 15, 16] * 4

stage0 = [0, 1, 2]*32
stage1 = stage0 + [3, 4]*64
stage2 = stage1 + [5, 6, 7, 8]*128
stage3 = stage2 + [9,10,11,12,13,14,15,16]*256
stage4 = stage3 + [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]*512
stage5 = stage4 + [
33,34,35,36,37,38,39,40,41,42,43,
44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
60,61,62,63,64
]*1024

# stage0 = [0, 1, 2, 3, 4, 5]
# stage1 = stage0 + [6, 7, 8, 9, 10, 11]
# stage2 = stage1 + [12, 13, 14, 15, 16, 17]

stages = [stage0, stage1, stage2, stage3, stage4, stage5]
sample_probs = [
  norm.cdf(2, 0, 4),
  norm.cdf(4, 0, 8) - norm.cdf(2, 0, 4),
  norm.cdf(8, 0, 16) - norm.cdf(4, 0, 8),
  norm.cdf(16, 0, 32) - norm.cdf(8, 0, 16),
  norm.cdf(32, 0, 64) - norm.cdf(16, 0, 32),
  1 - norm.cdf(32, 0, 64)
]

steps = 1000
cur_step = 0
start_mean = 0.0
interval = float(64/steps)

sorted_dict = {}

while cur_step < steps:
  cur_step += 1
  print(sample_probs)
  stage_sel = np.random.choice(a = [0,1,2,3,4,5], replace = True, p = sample_probs)
  sample_id = stages[stage_sel][np.random.randint(0, len(stages[stage_sel]))]

  if sample_id in sorted_dict:
    sorted_dict[sample_id] += 1
  else:
    sorted_dict[sample_id] = 1


  final_dict = sorted(sorted_dict.items(), key = lambda x: x[0])
  FrequencyBars(
    x = [x for (x, _) in final_dict],
    y = [y for (_, y) in final_dict],
  )

  start_mean += interval
  sample_probs = [
    norm.cdf(2, start_mean, 4),
    norm.cdf(4, start_mean, 8) - norm.cdf(2, start_mean, 4),
    norm.cdf(8, start_mean, 16) - norm.cdf(4, start_mean, 8),
    norm.cdf(16, start_mean, 32) - norm.cdf(8, start_mean, 16),
    norm.cdf(32, start_mean, 64) - norm.cdf(16, start_mean, 32),
    1 - norm.cdf(32, start_mean, 64)
  ]
  # input()