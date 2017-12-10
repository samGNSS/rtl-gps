import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class crossAmbiguity():
    def __init__(self, sampleRate, numFrequencyBins, numTimeBins, fftSize=1024):
        self.numTimeBins = numTimeBins
        self.sampleRate = sampleRate
        self.t = np.arange(0, numTimeBins) / self.sampleRate
        self.t = self.t[:, np.newaxis].T

        self.fftSize = fftSize
        self.freqReso = sampleRate / self.fftSize
        self.dopBins = np.arange(-numFrequencyBins // 2, numFrequencyBins // 2) * self.freqReso
        self.dopBins = self.dopBins[:, np.newaxis]
        self.freqMap = self._getFrequencyMap(self.t, self.dopBins)

    @staticmethod
    def _getFrequencyMap(t, dopBins):
        return np.exp(1j * 2 * np.pi * np.dot(dopBins, t))

    def process(self, in1, in2):
        in1 = in1[:self.numTimeBins] * self.freqMap
        in2 = in2[:self.numTimeBins]
        s = in1.shape
        ret = np.zeros((s[0], 2 * s[1] - 1), dtype=np.complex64)
        for row in range(ret.shape[0]):
            ret[row, :] = signal.fftconvolve(in1[row, :], np.conj(in2)[::-1])
        return ret
