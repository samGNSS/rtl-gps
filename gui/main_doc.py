from bokeh.plotting import figure, ColumnDataSource


def make_document(doc):
    source = ColumnDataSource({'x': [], 'y': [], 'color': []})

    def update():
        new = {}
        source.stream(new)

    doc.add_periodic_callback(update, 100)

    fig = figure(title='Streaming Circle Plot!', sizing_mode='scale_width',
                 x_range=[0, 1], y_range=[0, 1])

    fig.circle(source=source, x='x', y='y', color='color', size=10)

    doc.title = "Now with live updating!"
    doc.add_root(fig)
