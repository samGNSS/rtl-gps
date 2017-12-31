#!/home/sam/miniconda3/bin/ipython
from multiprocessing import Process, Lock, Value
from bokeh.layouts import gridplot
import numpy as np
from rtlsdr import RtlSdr

from gui.plotTypes import base_plot
from gui.rtl_gps_app import rtlGpsApp
from util.decorators import threadsafe

from dsp.cross_ambiguity import CrossAmbiguity
from dsp.detectors import CaCFAR

# get a global mutex lock
lock = Lock()
stopFlag = Value('b', False)

##############
# Parameters #
##############
center_freq = 960e6
samp_rate = 2.048e6
gain = 10
fft_size = 1024  # output size of fft, the input size is the samples_per_batch
waterfall_samples = 50  # number of rows of the waterfall
samples_in_time_plots = fft_size

sdr = RtlSdr()
sdr.center_freq = center_freq
sdr.sample_rate = samp_rate
sdr.freq_correction = 60
sdr.gain = 20

#DSP processors
cfar = CaCFAR(5, 10, 10)
# ambg = crossAmbiguity()


##############
# SET UP GUI #
##############

# Frequncy Sink (line plot)
fft_plot = base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency  Sink', disable_horizontal_zooming=True)
f = (np.linspace(-samp_rate / 2.0, samp_rate / 2.0, fft_size) + center_freq) / 1e6
fft_plot._input_buffer['y'] = np.zeros(fft_size)
fft_plot._input_buffer['T'] = np.zeros(fft_size)
fft_line = fft_plot.line(f, np.zeros(fft_size), color="aqua", line_width=1) # set x values but use dummy values for y
fftThres_line = fft_plot.line(f, np.zeros(fft_size), color="green", line_width=1)

# Time Sink (line plot)
time_plot = base_plot('Time [ms]', ' ', 'Time Sink', disable_horizontal_zooming=True)
t = np.linspace(0.0, samples_in_time_plots / samp_rate, samples_in_time_plots) * 1e3 # in ms
time_plot._input_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
timeI_line = time_plot.line(t, np.zeros(len(t)), color="aqua", line_width=1) # set x values but use dummy values for y
time_plot._input_buffer['q'] = np.zeros(samples_in_time_plots) # Q buffer (time domain)
timeQ_line = time_plot.line(t, np.zeros(len(t)), color="red", line_width=1) # set x values but use dummy values for y

# Waterfall Sink ("image" plot)
waterfall_plot = base_plot(' ', 'Time', 'Waterfall', disable_all_zooming=True)
waterfall_plot._set_x_range(0, fft_size) # Bokeh tries to automatically figure out range, but in this case we need to specify it
waterfall_plot._set_y_range(0, waterfall_samples)
waterfall_plot.axis.visible = False # i couldn't figure out how to update x axis when freq changes, so just hide them for now
waterfall_plot._input_buffer['waterfall'] = [np.ones((waterfall_samples, fft_size))*-100.0] # waterfall buffer, has to be in list form
waterfall_data = waterfall_plot.image(image = waterfall_plot._input_buffer['waterfall'],  # input
                                      x = 0, # start of x
                                      y = 0, # start of y
                                      dw = fft_size, # size of x
                                      dh = waterfall_samples)

# IQ/Constellation Sink ("circle" plot)
iq_plot = base_plot(' ', ' ', 'IQ Plot')
iq_plot._set_x_range(-1.0, 1.0) # this is to keep it fixed at -1 to 1. you can also just zoom out with mouse wheel and it will stop auto-ranging
iq_plot._set_y_range(-1.0, 1.0)
iq_plot._input_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
iq_plot._input_buffer['q'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
iq_data = iq_plot.circle(np.zeros(samples_in_time_plots),
                         np.zeros(samples_in_time_plots),
                         line_alpha=0.0, # setting line_width=0 didn't make it go away, but this works
                         fill_color="aqua",
                         fill_alpha=0.8, size=6) # size of circles

plots = gridplot([[fft_plot, time_plot], [waterfall_plot, iq_plot]], sizing_mode="scale_width", )


# This function gets called periodically, and is how the "real-time streaming mode" works
@threadsafe(lock)
def plot_update():
    timeI_line.data_source.data['y'] = time_plot._input_buffer['i'] # send most recent I to time sink
    timeQ_line.data_source.data['y'] = time_plot._input_buffer['q'] # send most recent Q to time sink
    iq_data.data_source.data = {'x': iq_plot._input_buffer['i'], 'y': iq_plot._input_buffer['q']} # send I and Q in one step using dict
    fft_line.data_source.data['y'] = fft_plot._input_buffer['y'] # send most recent psd to freq sink
    fftThres_line.data_source.data['y'] = fft_plot._input_buffer['T']
    waterfall_data.data_source.data['image'] = waterfall_plot._input_buffer['waterfall'] # send waterfall 2d array to waterfall sink


# Function that processes each batch of samples that comes in (currently, all DSP goes here)
@threadsafe(lock)
def process_samples(samples, _):
    # DSP
    PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
    thresh, _ = cfar.process(PSD)
    # thresh = np.zeros_like(PSD)
    
    # updating buffers
    waterfall = waterfall_plot._input_buffer['waterfall'][0] # pull waterfall from buffer
    waterfall[:] = np.roll(waterfall, -1, axis=0) # shifts waterfall 1 row
    waterfall[-1,:] = PSD # fill last row with new fft results
    waterfall_plot._input_buffer['waterfall'] = [waterfall]
    fft_plot._input_buffer['y'] = PSD
    fft_plot._input_buffer['T'] = thresh
    time_plot._input_buffer['i'] = samples.real
    time_plot._input_buffer['q'] = samples.imag
    iq_plot._input_buffer['i'] = samples.real
    iq_plot._input_buffer['q'] = samples.imag


# Function that runs asynchronous reading from the RTL, and is a blocking function
def start_sdr(flag):
    while True:
        if flag.value:
            break
        else:
            sdr.read_samples_async(process_samples, fft_size)
    print("Done")


# Start SDR sample processign as a separate thread
p = Process(target=start_sdr, args=(stopFlag,))
p.start()

################
# Assemble App #
################
myapp = rtlGpsApp() # start new pysdr app
myapp.makeDoc(plots, plot_update) # widgets, plots, periodic callback function, theme
myapp.makeBokehServer()
myapp.makeWebServer()

#run app
try:
    myapp.startServer() # start web server.  blocking
except KeyboardInterrupt:
    stopFlag.value = True
    p.join()
    sdr.close()

