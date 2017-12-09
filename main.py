#!/home/sam/miniconda3/bin/ipython

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from dsp.cross_ambiguity import crossAmbiguity


def main():
    sdr = RtlSdr()

    # configure device
    sdr.sample_rate = 1.048e6  # Hz
    sdr.center_freq = 92.5e6     # Hz
    sdr.freq_correction = 120   # PPM
    sdr.gain = 'auto'

    numSamps = 1024
    ambgFunc = crossAmbiguity(numSamps, 2.048e6, 100, 100)

    fig, axes = plt.subplots(4, 1)

    while True:
        iq = sdr.read_samples(numSamps)
        test = ambgFunc.process(iq, iq)

        axes[3].imshow(20 * np.log10(np.abs(test)), cmap='jet', aspect='auto')

        axes[0].plot(np.abs(iq), 'bs')
        axes[1].scatter(iq.real, iq.imag, c=np.abs(iq), cmap='jet')
        axes[2].plot(np.fft.fftshift(20 * np.log10(np.abs(np.fft.fft(iq)))))

        plt.pause(0.001)
        for ax in axes:
            ax.cla()


if __name__ == "__main__":
    main()
