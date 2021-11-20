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

def _get_generic_layout(title: str, x_name: str, y_name: str) -> go.Layout:
  return go.Layout(
    title = dict(text = title, font = dict(size = 38)),
    xaxis = dict(
      title = x_name, showgrid = False,
      showline = True, linecolor = 'black',
      mirror = True, linewidth = 2,
      tickfont = dict(size = 32), titlefont = dict(size = 38)
    ),
    yaxis = dict(
      title = y_name, showgrid = False,
      showline = True, linecolor = 'black',
      mirror = True, linewidth = 2,
      tickfont = dict(size = 32), titlefont = dict(size = 38)
    ),
  )

def _write_figure(fig       : go.Figure,
                  plot_name : str,
                  path      : pathlib.Path = None,
                  width     : int = None,
                  height    : int = None,
                  ) -> None:
  """
  Write plotly image & and html file if path exists.
  Otherwise only show html file.
  """
  if path:
    outf = lambda ext: str(path / "{}.{}".format(plot_name, ext))
    fig.write_html (outf("html"))
    fig.write_image(outf("png"), width = width, height = height)
  else:
    fig.show()
  return

def SingleScatterLine(x         : np.array,
                      y         : np.array,
                      title     : str,
                      x_name    : str,
                      y_name    : str,
                      plot_name : str,
                      path      : pathlib.Path = None,
                      ) -> None:
  """Plot a single line, with scatter points at datapoints."""
  layout = _get_generic_layout(title, x_name, y_name)
  fig = go.Figure(layout = layout)
  fig.add_trace(
    go.Scatter(
      x = x, y = y,
      mode = 'lines+markers',
      name = plot_name,
      showlegend = False,
      marker_color = "#00b3b3",
      opacity = 0.75
    )
  )
  _write_figure(fig, plot_name, path)
  return

def GroupScatterPlot(groups       : typing.Dict[str, typing.Dict[str, list]],
                     title        : str,
                     x_name       : str,
                     y_name       : str,
                     plot_name    : str,
                     marker_style : typing.List[str] = None,
                     path         : pathlib.Path = None,
                     ) -> None:
  """
  Plots groupped scatter plot of points in two-dimensional space.
  """
  layout = _get_generic_layout(title, x_name, y_name)
  # TODO update layout with this
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
  #   # margin={'l': 0, 'r': 0, 't': 70, 'b': 0},
  #   title = dict(text = title, font = dict(size = 38)),
  #   xaxis = dict(title = x_name, showgrid = False, showline = True, linecolor = 'black', mirror = True, linewidth = 2, tickfont = dict(size = 32), titlefont = dict(size = 38)),
  #   yaxis = dict(title = y_name, showgrid = False, showline = True, linecolor = 'black', mirror = True, linewidth = 2, tickfont = dict(size = 32), titlefont = dict(size = 38)),
  #   legend=dict(
  #     x=0.05,
  #     y=0.97,
  #     traceorder='normal',
  #     font=dict(size = 36,),
  #   )
  # )
  fig = go.Figure(layout = layout)
  if marker_style:
    if len(marker_style) != len(groups.keys()):
      raise ValueError("Mismatch between markers styles and number of groups")
    miter = iter(marker_style)
  else:
    miter = None
  for group, values in groups.items():
    feats = np.array(values['data'])
    names = values['names']
    fig.add_trace(
      go.Scatter(
        x = feats[:,0], y = feats[:,1],
        name = group,
        mode = 'markers',
        showlegend = True,
        opacity    = 1.0,
        marker     = next(miter) if miter else None,
        text       = names,
      )
    )
  _write_figure(fig, plot_name, path)
  return

def ScatterPlot(x         : np.array,
                y         : np.array,
                x_name    : str,
                y_name    : str,
                title     : str,
                plot_name : str,
                path      : pathlib.Path = None,
                ) -> None:
  """
  Implementation of a simple 2D scatter plot without groups.
  """
  layout = _get_generic_layout(title, x_name, y_name)
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  # )
  fig = go.Figure(layout = layout)

  fig.add_trace(
    go.Scatter(
      x = x,
      y = y,
      mode = 'markers',
      showlegend = True,
      opacity = 0.75,
    )
  )
  _write_figure(fig, plot_name, path)
  return

def FrequencyBars(x         : np.array,
                  y         : np.array,
                  title     : str,
                  x_name    : str,
                  plot_name : str,
                  path      : pathlib.Path = None,
                  ) -> None:
  """Plot frequency bars based on key."""
  layout = _get_generic_layout(title, x_name, "#of Occurences")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  #   xaxis = dict(title = x_name),
  #   yaxis = dict(title = "# of Occurences"),
  # )
  fig = go.Figure(layout = layout)
  fig.add_trace(
    go.Bar(
      x = x,
      y = y,
      showlegend = False,
      marker_color = '#ac3939',
      opacity = 0.75,
    )
  )
  _write_figure(fig, plot_name, path)
  return

def LogitsStepsDistrib(x              : typing.List[np.array],
                       atoms          : typing.List[str],
                       sample_indices : typing.List[str],
                       title          : str,
                       x_name         : str,
                       plot_name      : str,
                       path           : pathlib.Path = None,
                       ) -> None:
  """
  Categorical group-bar plotting.
  vocab_size number of groups. Groups are as many as prediction steps.
  Used to plot the probability distribution of BERT's token selection. 
  """
  layout = _get_generic_layout(title, x_name, "")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  #   xaxis = dict(title = x_name),
  #   # yaxis = dict(title = ""),
  # )
  fig = go.Figure(layout = layout)

  for pred, name in zip(x, sample_indices):
    fig.add_trace(
      go.Bar(
        name = name,
        x = atoms,
        y = pred,
      )
    )
  _write_figure(fig, plot_name, path)
  return

def GrouppedBars(groups    : typing.Dict[str, typing.Tuple[typing.List, typing.List]],
                 title     : str,
                 x_name    : str,
                 plot_name : str,
                 path      : pathlib.Path = None,
                 ) -> None:
  """
  Similar to LogitsStepsDistrib but more generic.
  Plots groups of bars.
  """
  # colors
  layout = _get_generic_layout(title, x_name, "")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = dict(text = title, font = dict(size = 26)),
  #   xaxis = dict(title = x_name, tickfont = dict(size = 24), titlefont = dict(size = 26)),
  #   # yaxis = dict(type = "log", gridcolor = '#c4c4c4', gridwidth = 0.4, tickformat = "0.1r", tickfont = dict(size = 24)),
  #   yaxis = dict(gridcolor = '#c4c4c4', gridwidth = 0.4, tickfont = dict(size = 24)),
  #   legend=dict(
  #     # x=0.1,
  #     # y=0.92,
  #     bgcolor = 'rgba(0,0,0,0)',
  #     traceorder='normal',
  #     font=dict(size = 24,),
  #   )
  # )
  fig = go.Figure(layout = layout)

  palette = itertools.cycle(px.colors.qualitative.T10)
  for group, (x, y) in groups.items():
    fig.add_trace(
      go.Bar(
        name = str(group),
        x = x,
        y = [(0.2+i if i == 0 else i) for i in y],
        marker_color = next(palette),
        textposition = "inside",
        text = ["" if i < 100 else "*" for i in y],
        textfont = dict(color = "white", size = 100),
      )
    )
  _write_figure(fig, plot_name, path)
  return

def CumulativeHistogram(x         : np.array,
                        y         : np.array,
                        title     : str,
                        x_name    : str,
                        plot_name : str,
                        path      : pathlib.Path = None,
                        ) -> None:
  """Plot percent cumulative histogram."""
  layout = _get_generic_layout(title, x_name, "% of Probability Density")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  #   xaxis = dict(title = x_name),
  #   yaxis = dict(title = "% of Probability Density"),
  # )
  fig = go.Figure(layout = layout)
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
  _write_figure(fig, plot_name, path)
  return

def NormalizedRadar(r         : np.array,
                    theta     : typing.List[str],
                    title     : str,
                    plot_name : str,
                    path      : pathlib.Path = None,
                    ) -> None:
  """Radar chart for feature plotting"""
  layout = _get_generic_layout(title, "", "")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  # )
  fig = go.Figure(layout = layout)
  fig.add_trace(
    go.Scatterpolar(
      r = r,
      theta = theta,
      fill = 'toself',
      marker_color = "#cbef0e",
    )
  )
  _write_figure(fig, plot_name, path)
  return

def CategoricalViolin(x         : np.array,
                      y         : typing.List[np.array],
                      title     : str,
                      x_name    : str,
                      plot_name : str,
                      path      : pathlib.Path = None,
                      ) -> None:
  """Plot percent cumulative histogram."""
  layout = _get_generic_layout(title, x_name, "Distribution / category")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  #   violingap = 0,
  #   violinmode = 'overlay',
  #   xaxis = dict(title = x_name),
  #   yaxis = dict(title = "Distribution / category"),
  # )
  fig = go.Figure(layout = layout)
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
  _write_figure(fig, plot_name, path)
  return

def RelativeDistribution(x         : np.array,
                         y         : typing.List[np.array],
                         title     : str,
                         x_name    : str,
                         plot_name : str,
                         path      : pathlib.Path = None,
                         ) -> None:
  """Plot smoothened relative distribution of data"""
  layout = _get_generic_layout(title, x_name, "")
  # layout = go.Layout(
  #   plot_bgcolor='rgba(0,0,0,0)',
  #   title = title,
  #   xaxis = dict(title = x_name, tickfont = dict(size = 28), titlefont = dict(size = 30)),
  #   yaxis = dict(gridcolor = '#c4c4c4', gridwidth = 0.4, tickfont = dict(size = 28)),
  #   legend=dict(
  #     x=0.75,
  #     y=0.75,
  #     bgcolor = 'rgba(0,0,0,0)',
  #     traceorder='normal',
  #     font=dict(size = 28,),
  #   )
  # )
  fig = ff.create_distplot(
    y,
    x,
    curve_type = 'normal',
    histnorm = "probability",
    show_rug = False,
    bin_size = 1,
    show_hist = True
  )
  fig.update_layout(layout)
  _write_figure(fig, plot_name, path)
  return