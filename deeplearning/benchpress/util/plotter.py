# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
In-house plotter module that plots data.
Based on plotly module
"""
import typing
import pathlib
import numpy as np
import itertools

from plotly import graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px

from deeplearning.benchpress.util import logging as l

example_formats = """
  margin = {'l': 0, 'r': 0, 't': 0, 'b': 0}   # Eliminates excess background around the plot (Can hide the title)
  plot_bgcolor = 'rgba(0,0,0,0)' or "#000fff" # Sets the background color of the plot
"""

def _get_generic_figure(**kwargs) -> go.Layout:
  """
  Constructor of a basic plotly layout.
  All keyword arguments are compatible with plotly documentation

  Exceptions:
    axisfont instead of titlefont. Reserved titlefont for title's 'font' size property.
  """
  # Title and axis names
  title  = kwargs.get('title', "")
  x_name = kwargs.get('x_name', "")
  y_name = kwargs.get('y_name', "")

  # Font sizes
  titlefont   = kwargs.get('titlefont', None)
  axisfont    = kwargs.get('axisfont',  None)
  tickfont    = kwargs.get('tickfont',  None)

  # Plot line and axis options
  showline_x  = kwargs.get('showline_x',  None)
  showline_y  = kwargs.get('showline_y',  None)
  linecolor   = kwargs.get('linecolor',   None)
  gridcolor_x = kwargs.get('gridcolor_x', None)
  gridcolor_y = kwargs.get('gridcolor_y', None)
  mirror      = kwargs.get('mirror',      None)
  showgrid_x  = kwargs.get('showgrid_x',  None)
  showgrid_y  = kwargs.get('showgrid_y',  None)
  linewidth   = kwargs.get('linewidth',   None)
  gridwidth   = kwargs.get('gridwidth',   None)
  margin      = kwargs.get('margin', {'l': 40, 'r': 45, 't': 95, 'b': 0})
  x_tickangle = kwargs.get('x_tickangle', None)
  showticklabels_x = kwargs.get('showticklabels_x', True)
  showticklabels_y = kwargs.get('showticklabels_y', True)

  # Legend
  legend_x   = kwargs.get('legend_x', 1.0)
  legend_y   = kwargs.get('legend_y', 1.0)
  traceorder = kwargs.get('traceorder', None)
  legendfont = kwargs.get('legendfont', None)

  # Background
  bg_color = kwargs.get('bg_color', None)

  # Violin options
  violingap  = kwargs.get('violingap',  None)
  violinmode = kwargs.get('violinmode', None)

  title = dict(text = title, font = dict(size = titlefont))
  yaxis = dict(
             title     = y_name,     showgrid = showgrid_y,
             showline  = showline_y, linecolor = linecolor,
             mirror    = mirror,     linewidth = linewidth,
             gridwidth = gridwidth,
             gridcolor = gridcolor_y,
             tickfont  = dict(size = tickfont),
             titlefont = dict(size = axisfont),
             showticklabels = showticklabels_y,
             rangemode="tozero"
          )
  xaxis = dict(
            title     = x_name,     showgrid = showgrid_x,
            showline  = showline_x, linecolor = linecolor,
            mirror    = mirror,     linewidth = linewidth,
            gridcolor = gridcolor_x,
            tickfont  = dict(size = tickfont),
            titlefont = dict(size = axisfont),
            showticklabels = showticklabels_x,
          )
  layout = go.Layout(
    plot_bgcolor = bg_color,
    margin       = margin,
    legend       = dict(x = legend_x, y = legend_y, traceorder = traceorder, font = dict(size = legendfont)),
    title        = title,
    xaxis        = xaxis,
    yaxis        = yaxis,
    violingap    = violingap,
    violinmode   = violinmode,
  )
  fig = go.Figure(layout = layout)
  if x_tickangle:
    fig.update_xaxes(tickangle = 45)
  fig.update_yaxes(automargin = True)
  return fig

def _write_figure(fig       : go.Figure,
                  plot_name : str,
                  path      : pathlib.Path = None,
                  **kwargs
                  ) -> None:
  """
  Write plotly image & and html file if path exists.
  Otherwise only show html file.
  """
  if path:
    path.mkdir(parents = True, exist_ok = True)
    outf = lambda ext: str(path / "{}.{}".format(plot_name, ext))
    try:
      fig.write_html (outf("html"))
    except ValueError:
      l.logger().warn("HTML plot failed", ddp_nodes = True)
    try:
      fig.write_image(outf("png"), width = kwargs.get('width', 1024), height = kwargs.get('height', 768))
    except ValueError:
      l.logger().warn("PNG plot failed", ddp_nodes = True)
    try:
      fig.write_image(outf("pdf"))
    except ValueError:
      l.logger().warn("PDF plot failed", ddp_nodes = True)
  else:
    fig.show()
  return

def SingleScatterLine(x         : np.array,
                      y         : np.array,
                      plot_name : str,
                      path      : pathlib.Path = None,
                      **kwargs,
                      ) -> None:
  """Plot a single line, with scatter points at datapoints."""
  fig = _get_generic_figure(**kwargs)
  fig.add_trace(
    go.Scatter(
      x = x, y = y,
      mode = kwargs.get('mode', 'lines+markers'),
      name = plot_name,
      showlegend = kwargs.get('showlegend', False),
      marker_color = kwargs.get('marker_color', "#00b3b3"),
      opacity = kwargs.get('opacity', 0.75),
    )
  )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def MultiScatterLine(x         : typing.List[np.array],
                     y         : typing.List[np.array],
                     names     : typing.List[str],
                     plot_name : str,
                     path      : pathlib.Path = None,
                     **kwargs,
                     ) -> None:
  """
  Implementation of a simple, ungroupped 2D plot of multiple scatter lines.
  """
  fig = _get_generic_figure(**kwargs)
  for xx, yy, n in zip(x, y, names):
    fig.add_trace(
      go.Scatter(
        x = xx,
        y = yy,
        name = n,
        mode = kwargs.get('mode', 'lines+markers'),
        showlegend = kwargs.get('showlegend', True),
        opacity = kwargs.get('opacity', 0.75),
      )
    )
    _write_figure(fig, plot_name, path, **kwargs)
  return

def GroupScatterPlot(groups       : typing.Dict[str, typing.Dict[str, list]],
                     plot_name    : str,
                     marker_style : typing.List[str] = None,
                     path         : pathlib.Path = None,
                     **kwargs,
                     ) -> None:
  """
  Plots groupped scatter plot of points in two-dimensional space.
  Groups must comply to the following format:

  groups = {
    'group_name': {
      'data': [[xi, yi], ...],
      'names': [ni, ...],
      'frequency': [1, 1, 2, ...] if you want scatter bubbles based on frequency.
    }
  }
  """
  fig = _get_generic_figure(**kwargs)
  if marker_style:
    if len(marker_style) != len(groups.keys()):
      raise ValueError("Mismatch between markers styles and number of groups")
    miter = iter(marker_style)
  else:
    miter = None
  for group, values in groups.items():
    if len(values['data']) == 0:
      continue
    feats = np.array(values['data'])
    names = values['names'] 
    fig.add_trace(
      go.Scatter(
        x = feats[:,0], y = feats[:,1],
        name = group,
        mode = kwargs.get('mode', 'markers'),
        showlegend  = kwargs.get('showlegend', True),
        opacity     = kwargs.get('opacity', 1.0),
        marker      = next(miter) if miter else ({'size': 18} if 'frequency' not in values else None),
        marker_size = [14*x for x in values['frequency']] if 'frequency' in values else 18,
        text        = names,
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def SliderGroupScatterPlot(steps        : typing.List[typing.Dict[str, typing.Dict[str, list]]],
                           plot_name    : str,
                           path         : pathlib.Path = None,
                           **kwargs,
                           ) -> None:
  fig = _get_generic_figure(**kwargs)
  print(len(steps.keys()))
  for step, step_data in steps.items():
    for group, values in step_data.items():
      if len(values['data']) == 0:
        fig.add_trace(
          go.Scatter(
            visible = False,
            x = [], y = [],
            name = group,
          )
        )
      else:
        feats = np.array(values['data'])
        names = values['names']
        fig.add_trace(
          go.Scatter(
            visible = False,
            x = feats[:,0], y = feats[:,1],
            name = group,
            mode = kwargs.get('mode', 'markers'),
            showlegend  = kwargs.get('showlegend', True),
            opacity     = kwargs.get('opacity', 1.0),
            marker_size = [14*x for x in values['frequency']] if 'frequency' in values else None,
            text        = names,
          )
        )
  print(len(fig.data))
  fig.data[0].visible = True
  fig.data[1].visible = True
  fig.data[2].visible = True
  fig.data[3].visible = True
  fig.data[4].visible = True
  slider_steps = []
  for i in range(len(steps.keys())):
    step = dict(
      method = 'update',
      args = [
        {'visible': [False] * len(fig.data)},
        {"title": "Users with total predictions >= {}".format(i+2)}
      ]
    )
    step["args"][0]["visible"][i*5] = True
    step["args"][0]["visible"][i*5 + 1] = True
    step["args"][0]["visible"][i*5 + 2] = True
    step["args"][0]["visible"][i*5 + 3] = True
    step["args"][0]["visible"][i*5 + 4] = True
    slider_steps.append(step)
  sliders = [dict(steps = slider_steps)]
  fig.update_layout(sliders = sliders)
  _write_figure(fig, plot_name, path, **kwargs)
  return


def FrequencyBars(x         : np.array,
                  y         : np.array,
                  plot_name : str,
                  path      : pathlib.Path = None,
                  vlines    : typing.List[typing.Tuple[int, str]] = None,
                  **kwargs,
                  ) -> None:
  """Plot frequency bars based on key."""
  fig = _get_generic_figure(**kwargs)
  fig.add_trace(
    go.Bar(
      x = x,
      y = y,
      showlegend = False,
      marker_color = '#ac3939',
      opacity = 0.75,
    )
  )
  if vlines:
    for (vline, annotation) in vlines:
      fig.add_vline(
        x = vline,
        line_width          = 1,
        line_dash           = "solid",
        line_color          = "black",
        annotation_position = "bottom",
        annotation_text     = annotation,
      )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def LogitsStepsDistrib(x              : typing.List[np.array],
                       atoms          : typing.List[str],
                       sample_indices : typing.List[str],
                       plot_name      : str,
                       path           : pathlib.Path = None,
                       **kwargs,
                       ) -> None:
  """
  Categorical group-bar plotting.
  vocab_size number of groups. Groups are as many as prediction steps.
  Used to plot the probability distribution of BERT's token selection. 
  """
  fig = _get_generic_figure(**kwargs)

  for pred, name in zip(x, sample_indices):
    fig.add_trace(
      go.Bar(
        name = name,
        x = atoms,
        y = pred,
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def GrouppedBars(groups    : typing.Dict[str, typing.Tuple[typing.List, typing.List]],
                 plot_name : str,
                 text      : typing.List[str] = None,
                 path      : pathlib.Path = None,
                 **kwargs,
                 ) -> None:
  """
  Similar to LogitsStepsDistrib but more generic.
  Plots groups of bars.

  Groups must comply to the following format:
  groups = {
    'group_name': ([], [])
  }
  """
  # colors
  fig = _get_generic_figure(**kwargs)

  palette = itertools.cycle(px.colors.qualitative.T10)
  for group, (x, y) in groups.items():
    fig.add_trace(
      go.Bar(
        name = str(group),
        x = x,
        y = [(0.2+i if i == 0 else i) for i in y],
        marker_color = next(palette),
        textposition = kwargs.get('textposition', 'inside'),
        # text = text,
        text = ["" if i < 100 else "*" for i in y],
        textfont = dict(color = "white", size = 140),
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def SliderGrouppedBars(steps     : typing.Dict[int, typing.Dict[str, typing.Tuple[typing.List, typing.List]]],
                       plot_name : str,
                       text      : typing.List[str] = None,
                       path      : pathlib.Path = None,
                       **kwargs,
                       ) -> None:
  """
  Similar to LogitsStepsDistrib but more generic.
  Plots groups of bars.

  Groups must comply to the following format:
  groups = {
    'group_name': ([], [])
  }
  """
  # colors
  fig = _get_generic_figure(**kwargs)

  for step_id, step_data in steps.items():
    palette = itertools.cycle(px.colors.qualitative.T10)
    print(step_id)
    print(step_data.keys())
    for group, (x, y) in step_data.items():
      fig.add_trace(
        go.Bar(
          visible = False,
          name = str(group),
          x = x,
          y = [(0.0+i if i == 0 else i) for i in y],
          marker_color = next(palette),
          textposition = kwargs.get('textposition', 'inside'),
          text = text,
          # text = ["" if i < 100 else "*" for i in y],
          textfont = dict(color = "white", size = 140),
        )
      )
  print(len(fig.data))
  fig.data[0].visible = True
  slider_steps = []
  for i in range(len(steps.keys())):
    step = dict(
      method = 'update',
      args = [
        {'visible': [False] * len(fig.data)},
        {"title": "Users with total predictions >= {}".format(i+2)}
      ]
    )
    step["args"][0]["visible"][i] = True
    slider_steps.append(step)
  sliders = [dict(steps = slider_steps)]
  fig.update_layout(sliders = sliders)
  _write_figure(fig, plot_name, path, **kwargs)
  return

def CumulativeHistogram(x         : np.array,
                        y         : np.array,
                        plot_name : str,
                        path      : pathlib.Path = None,
                        **kwargs,
                        ) -> None:
  """Plot percent cumulative histogram."""
  fig = _get_generic_figure(**kwargs)
  fig.add_trace(
    go.Histogram(
      x = x,
      y = y,
      xbins = dict(size = 8),
      cumulative_enabled = True,
      histfunc = 'sum',
      histnorm = 'percent',
      showlegend = False,
      marker_color = '#1d99a3',
      opacity = 0.65,
    )
  )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def NormalizedRadar(r         : np.array,
                    theta     : typing.List[str],
                    plot_name : str,
                    path      : pathlib.Path = None,
                    **kwargs,
                    ) -> None:
  """Radar chart for feature plotting"""
  fig = _get_generic_figure(**kwargs)
  fig.add_trace(
    go.Scatterpolar(
      r = r,
      theta = theta,
      fill = 'toself',
      marker_color = "#cbef0e",
    )
  )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def GrouppedRadar(groups    : typing.Dict[str, typing.Tuple[typing.List[float], typing.List[str]]],
                  plot_name : str,
                  path      : pathlib.Path = None,
                  **kwargs,
                  ) -> None:
  """
  Plot multiple groups within the same radar.
  Input format of groups:
  groups = {
    'group1': [
      [v1, v2, v3], # float values
      [th1, th2, th3] # axis names
    ]
  }
  """
  # fig = _get_generic_figure(**kwargs)
  fig = go.Figure(
    layout = go.Layout(title = kwargs.get('title'))
  )
  for group, (vals, thetas) in groups.items():
    fig.add_trace(
      go.Scatterpolar(
        r = vals + [vals[0]],
        theta = thetas + [thetas[0]],
        name = group,
      )
    )
  # fig = px.line_polar(df, r='r', theta='theta', line_close=True)
  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        # range=[0, 1]
      )),
        showlegend=True
  )
  _write_figure(fig, plot_name, path, width = 1800, height = 1800, **kwargs)
  return

def CategoricalViolin(x         : np.array,
                      y         : typing.List[np.array],
                      plot_name : str,
                      path      : pathlib.Path = None,
                      **kwargs,
                      ) -> None:
  """Plot percent cumulative histogram."""
  fig = _get_generic_figure(**kwargs)
  for xel, yel in zip(x, y):
    fig.add_trace(
      go.Violin(
        x = [xel]*len(yel),
        y = yel,
        name = xel,
        # side = 'positive',
        meanline_visible = True,
        box_visible = True,
        showlegend = False,
        opacity = 0.65,
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def GrouppedViolins(data      : typing.Dict[str, typing.Dict[str, typing.List[int]]],
                    plot_name : str,
                    path      : pathlib.Path = None,
                    **kwargs,
                    ) -> None:
  """
  Plot groupped violins. Inputs must be:
  data = {
    'group_name': {
      'feat_1': [v1, v2, v3...],
      'feat_2': [v1, v2, v3...],
    }
  } etc.
  """
  fig = _get_generic_figure(**kwargs)
  # fig = go.Figure(layout = go.Layout(violinmode = 'group'))
  for group in data.keys():
    fig.add_trace(
      go.Violin(
        x = [x for y in [[x]*len(data[group][x]) for x in data[group].keys()] for x in y],
        y = [x for y in data[group].values() for x in y],
        legendgroup = group,
        scalegroup  = group,
        name        = group,
      )
    )
  _write_figure(fig, plot_name, path, **kwargs)
  return

def RelativeDistribution(x         : np.array,
                         y         : typing.List[np.array],
                         plot_name : str,
                         path      : pathlib.Path = None,
                         **kwargs,
                         ) -> None:
  """Plot smoothened relative distribution of data"""
  # layout = _get_generic_layout(**kwargs)
  fig = ff.create_distplot(
    y,
    x,
    curve_type = 'normal',
    histnorm = "probability",
    show_rug = kwargs.get('show_rug', False),
    bin_size = kwargs.get('bin_size', 1),
    show_hist = True
  )
  # fig.update_layout(layout)
  _write_figure(fig, plot_name, path, **kwargs)
  return
