#!/home/sam/miniconda3/bin/ipython

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from dsp.cross_ambiguity import crossAmbiguity
from gui.iq_streamer import iqStream

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource


def main(args):
    sdr = RtlSdr()

    # configure device
    sdr.sample_rate = args.rate
    sdr.center_freq = args.freq
    sdr.freq_correction = args.freqCorr
    sdr.gain = args.gain

    iqStreamer = iqStream(sdr)

    numSamps = 1024
    ambgFunc = crossAmbiguity(args.rate, 100, 100)

    def make_document(doc):
        source = ColumnDataSource({'iq': [], 'iqI': [], 'iqQ': []})

        def update():
            iq = iqStreamer.sample(numSamps)
            print(iq)
            new = {'iq': iq, 'iqI': iq.real, 'iqQ': iq.imag}
            source.stream(new)

        doc.add_periodic_callback(update, 100)

        fig = figure(title='Streaming Iq plot!', sizing_mode='scale_width',
                     x_range=[-1, 1], y_range=[-1, 1])

        fig.circle(source=source, x='iqI', y='iqQ', color='blue', size=10)

        doc.title = "Now with live updating!"
        doc.add_root(fig)

    apps = {'/': Application(FunctionHandler(make_document))}

    server = Server(apps, port=5006)
    print("Starting")
    server.start()

    # fig, axes = plt.subplots(4, 1)
    #
    # while True:
    #     iq = sdr.read_samples(numSamps)
    #
    #     test = ambgFunc.process(iq, iq)
    #
    #     axes[3].imshow(20 * np.log10(np.abs(test)), cmap='jet', aspect='auto')
    #
    #     axes[0].plot(np.abs(iq), 'bs')
    #     axes[1].scatter(iq.real, iq.imag, c=np.abs(iq), cmap='jet')
    #     # axes[2].plot(np.fft.fftshift(20 * np.log10(np.abs(np.fft.fft(iq)))))
    #     axes[2].plot(np.abs(test[50, :]))
    #
    #     plt.pause(0.0001)
    #     for ax in axes:
    #         ax.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steam samples from an RtlSdr')
    parser.add_argument('-s', '--rate', type=float, default=2.048e6,
                        help='Sample Rate (Sps)')
    parser.add_argument('-f', '--freq', type=float, default=92.5e6,
                        help='Frequency (Hz)')
    parser.add_argument('-g', '--gain', type=float, default=20,
                        help='Gain (dB)')
    parser.add_argument('-c', '--freqCorr', type=float, default=120,
                        help='Frequency Correction (ppm)')

    args = parser.parse_args()
    main(args)
