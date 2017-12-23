"""
Collection of plot types
"""

from bokeh.plotting import Figure
from bokeh.models import WheelZoomTool, BoxZoomTool, ResetTool, SaveTool # all the tools we want- reference http://bokeh.pydata.org/en/0.10.0/docs/reference/models/tools.html
from bokeh.models import Range1d
from multiprocessing import Manager


def base_plot(x_label, y_label, title, **kwargs):

    # This allows disabling of horizontal zooming, which gets annoying in most dsp plots
    # e.g. usage: fft_plot = pysdr.base_plot('Freq', 'PSD', 'Frequency', disable_horizontal_zooming=True)
    if 'disable_horizontal_zooming' in kwargs and kwargs['disable_horizontal_zooming']: # if it's specified and is set True
        tools = [WheelZoomTool(dimensions='height')]
    else:
        tools = [WheelZoomTool()]

    # Similar to above, except disable all zooming, perfect for waterfall plots
    if 'disable_all_zooming' in kwargs and kwargs['disable_all_zooming']: # if it's specified and is set True
        tools = []  # removes the WheelZoomTool we just added above

    if 'plot_height' in kwargs:
        plot_height = kwargs['plot_height']
    else:
        plot_height = 200

    # Create the Bokeh figure
    plot = Figure(plot_width = 200, # this is more for the ratio, because we have auto-width scaling
                  plot_height = plot_height,
                  y_axis_label = y_label,
                  x_axis_label = x_label,
                  tools = tools + [BoxZoomTool(), ResetTool(), SaveTool()], # all the other tools we want- reference http://bokeh.pydata.org/en/0.10.0/docs/reference/models/tools.html
                  title = title)  # use min_border=30 to add padding between plots, if we ever want it

    # sets wheel zoom active by default (tools[0] is the wheelzoom), unless zooming was disabled
    if 'disable_all_zooming' not in kwargs:
        plot.toolbar.active_scroll = plot.toolbar.tools[0]

    # hides stupid bokeh logo
    plot.toolbar.logo = None

    # add more intuitive functions to set x and y ranges
    def _set_x_range(min_x, max_x): # without the underscore it wont work, bokeh/core/has_props.py overloads __setattr__ to intercept attribute setting that is not private
        plot.x_range = Range1d(min_x, max_x)
    def _set_y_range(min_y, max_y):
        plot.y_range = Range1d(min_y, max_y)
    plot._set_x_range = _set_x_range # add functions to object
    plot._set_y_range = _set_y_range

    # Add input buffer
    manager = Manager()
    plot._input_buffer = manager.dict()

    # return the bokeh figure object
    return plot
