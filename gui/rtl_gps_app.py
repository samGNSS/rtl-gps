from flask import Flask, render_template

from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import autoload_server
from bokeh.server.server import Server
from bokeh.util.browser import view # utility to Open a browser to view the specified location.

from multiprocessing import Process, Manager

# from pysdr.themes import black_and_white # removed relative imports to work with python3


"""
Bokeh app
Look at: https://github.com/pysdr/pysdr/blob/master/pysdr/pysdr_app.py
"""


class rtlGpsApp():
    def __init__(self):
        self.mainApp = None
        self.flask_app = Flask('__main__')

        # GET routine for root page
        @self.flask_app.route('/', methods=['GET'])
        def bkapp_page():
            script = autoload_server(url='http://localhost:5006/bkapp')
            return render_template('index.html', script=script)

    def makeDoc(self, plots, plotUpdater, widgets = None, theme = None, updateTime_ms = 100):
        print("Hello")

        def docCallback(doc):
            if widgets is not None:
                doc.add_root(widgets)

            if theme is not None:
                doc.theme = theme

            doc.add_root(plots)
            doc.add_periodic_callback(plotUpdater, updateTime_ms)

        self.mainApp = Application(FunctionHandler(docCallback))

    def makeBokehServer(self):
        self.io_loop = IOLoop.current() # creates an IOLoop for the current thread
        # Create the Bokeh server, which "instantiates Application instances as clients connect".  We tell it the bokeh app and the ioloop to use
        server = Server({'/bkapp': self.mainApp}, io_loop=self.io_loop, allow_websocket_origin=["localhost:8080"])
        server.start()


    def makeWebServer(self):
        # Create the web server using tornado (separate from Bokeh server)
        print('Opening Flask app with embedded Bokeh application on http://localhost:8080/')
        http_server = HTTPServer(WSGIContainer(self.flask_app)) # A non-blocking, single-threaded HTTP server. serves the WSGI app that flask provides. WSGI was created as a low-level interface between web servers and web applications or frameworks to promote common ground for portable web application development
        http_server.listen(8080) # this is the single-process version, there are multi-process ones as well
        # Open browser to main page
        self.io_loop.add_callback(view, "http://localhost:8080/") #

    def startServer(self):
        self.io_loop.start()

if __name__ == "__main__":
    test = rtlGpsApp()
