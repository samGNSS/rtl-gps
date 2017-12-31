import numpy as np
import scipy.signal as signal


class CrossAmbiguity():
    """Calculates the cross ambiguity function between two vectors
    """
    def __init__(self, sampleRate, numFrequencyBins, numTimeBins, fftSize=1024):
        self.numTimeBins = numTimeBins
        self.sampleRate = sampleRate
        self.time = np.arange(0, numTimeBins) / self.sampleRate
        self.time = self.time[:, np.newaxis].T

        self.fftSize = fftSize
        self.freqReso = sampleRate / self.fftSize
        self.dopBins = np.arange(-numFrequencyBins // 2, numFrequencyBins // 2) * self.freqReso
        self.dopBins = self.dopBins[:, np.newaxis]
        self.freqMap = self._getFrequencyMap(self.time, self.dopBins)

    @staticmethod
    def _getFrequencyMap(time, dopBins):
        return np.exp(1j * 2 * np.pi * np.dot(dopBins, time))

    def process(self, in1, in2):
        """Process the inputs
        """
        in1 = in1[:self.numTimeBins] * self.freqMap
        in2 = in2[:self.numTimeBins]
        ret = signal.fftconvolve(in1, np.conj(in2)[np.newaxis, ::-1])
        return ret
