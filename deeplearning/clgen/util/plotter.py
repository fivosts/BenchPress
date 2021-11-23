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

example_formats = """
  margin = {'l': 0, 'r': 0, 't': 0, 'b': 0}   # Eliminates excess background around the plot (Can hide the title)
  plot_bgcolor = 'rgba(0,0,0,0)' or "#000fff" # Sets the background color of the plot
"""

def _get_generic_layout(**kwargs) -> go.Layout:
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
  titlefont = kwargs.get('titlefont', 38)
  axisfont  = kwargs.get('axisfont', 38)
  tickfont  = kwargs.get('tickfont', 32)

  # Plot line and axis options
  showline  = kwargs.get('showline',  True)
  linecolor = kwargs.get('linecolor', 'black')
  gridcolor = kwargs.get('gridcolor', "#eee")
  mirror    = kwargs.get('mirror',    False)
  showgrid  = kwargs.get('showgrid',  True)
  linewidth = kwargs.get('linewidth', 2)
  gridwidth = kwargs.get('gridwidth', 1)
  margin    = kwargs.get('margin', {'l': 80, 'r': 80, 't': 100, 'b': 80})

  # Legend
  legend_x   = kwargs.get('legend_x', 1.02)
  legend_y   = kwargs.get('legend_y', 1.0)
  traceorder = kwargs.get('traceorder', "normal")
  legendfont = kwargs.get('legendfont', 24)

  # Background
  plot_bgcolor = kwargs.get('plot_bgcolor', "#fff")

  # Violin options
  violingap  = kwargs.get('violingap', 0)
  violinmode = kwargs.get('violinmode', 'overlay')

  title = dict(text = title, font = dict(size = titlefont))
  yaxis = dict(
             title     = y_name,   showgrid = showgrid,
             showline  = showline, linecolor = linecolor,
             mirror    = mirror,   linewidth = linewidth,
             gridwidth = gridwidth,
             tickfont  = dict(size = tickfont),
             titlefont = dict(size = axisfont)
          )
  xaxis = dict(
            title     = x_name,   showgrid = showgrid,
            showline  = showline, linecolor = linecolor,
            mirror    = mirror,   linewidth = linewidth,
            tickfont  = dict(size = tickfont),
            titlefont = dict(size = axisfont)
          )
  return go.Layout(
    plot_bgcolor = plot_bgcolor,
    margin       = margin,
    legend       = dict(x = legend_x, y = legend_y, traceorder = traceorder, legendfont = legendfont),
    title        = title,
    xaxis        = xaxis,
    yaxis        = yaxis,
    violingap    = violingap,
    violinmode   = violinmode,
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
                      plot_name : str,
                      path      : pathlib.Path = None,
                      **kwargs,
                      ) -> None:
  """Plot a single line, with scatter points at datapoints."""
  layout = _get_generic_layout(title = title, x_name = x_name, y_name = y_name, kwargs)
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
                     plot_name    : str,
                     marker_style : typing.List[str] = None,
                     path         : pathlib.Path = None,
                     **kwargs,
                     ) -> None:
  """
  Plots groupped scatter plot of points in two-dimensional space.
  """
  layout = _get_generic_layout(
    title = title,
    x_name = x_name,
    y_name = y_name,
    kwargs,
    # plot_bgcolor = 'rbga(0,0,0,0)',
    # margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    # legend_x = 0.05,
    # legend_y = 0.97,
  )
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
                plot_name : str,
                path      : pathlib.Path = None,
                **kwargs,
                ) -> None:
  """
  Implementation of a simple 2D scatter plot without groups.
  """
  layout = _get_generic_layout(
    title = title,
    x_name = x_name,
    y_name = y_name,
    # plot_bgcolor = 'rgba(0,0,0,0)'
    kwargs,
  )
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
                  plot_name : str,
                  path      : pathlib.Path = None,
                  **kwargs,
                  ) -> None:
  """Plot frequency bars based on key."""
  layout = _get_generic_layout(title = title, x_name = x_name, y_name = "#of Occurences", kwargs)
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
                       plot_name      : str,
                       path           : pathlib.Path = None,
                       **kwargs,
                       ) -> None:
  """
  Categorical group-bar plotting.
  vocab_size number of groups. Groups are as many as prediction steps.
  Used to plot the probability distribution of BERT's token selection. 
  """
  layout = _get_generic_layout(
    title = title,
    x_name = x_name,
    kwargs
  )
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
                 plot_name : str,
                 path      : pathlib.Path = None,
                 **kwargs,
                 ) -> None:
  """
  Similar to LogitsStepsDistrib but more generic.
  Plots groups of bars.
  """
  # colors
  layout = _get_generic_layout(
    title = title,
    x_name = x_name,
    # plot_bgcolor = 'rgba(0,0,0,0)',
    # gridwidth = 0.4,
    # gridcolor = "#c4c4c4",
    # legend_x = 0.1,
    # legend_y = 0.92,
    kwargs,
  )
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
                        plot_name : str,
                        path      : pathlib.Path = None,
                        **kwargs,
                        ) -> None:
  """Plot percent cumulative histogram."""
  layout = _get_generic_layout(
    title = title,
    x_name = x_name,
    y_name = "% of Probability Density",
    kwargs
  )
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
                    plot_name : str,
                    path      : pathlib.Path = None,
                    **kwargs,
                    ) -> None:
  """Radar chart for feature plotting"""
  layout = _get_generic_layout(title = title, kwargs)
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
                      plot_name : str,
                      path      : pathlib.Path = None,
                      **kwargs,
                      ) -> None:
  """Plot percent cumulative histogram."""
  layout = _get_generic_layout(
    title      = title,
    x_name     = x_name,
    # y_name     = "Distribution / category",
    # violingap  = 0,
    # violinmode = 'overlay',
    kwargs,
  )
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
                         plot_name : str,
                         path      : pathlib.Path = None,
                         **kwargs,
                         ) -> None:
  """Plot smoothened relative distribution of data"""
  layout = _get_generic_layout(
    title  = title,
    x_name = x_name,
    # plot_bgcolor = 'rgba(0,0,0,0)',
    # gridwidth = 0.4,
    # gridcolor = "#c4c4c4",
    # legend_x  = 0.75,
    # legend_y  = 0.75,
    # traceorder = 'normal',
    kwargs,
  )
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