import numpy as np
import scipy.signal as signal
import warnings


class BaseFilter():
    """FIR filter base class

    Simple class to manage an FIR filter
    """
    def __init__(self, sampRate, cutoffFreq, windowType="blackman", numTaps=512):
        self._sampRate = sampRate
        self._cutoffFreq = cutoffFreq
        self._cutoffFreqNorm = 2 * self._cutoffFreq/self._sampRate
        self._windowType = windowType
        self._numTaps = numTaps

        #design the filter
        self._taps = self._designFilter()

    # properties
    @property
    def sampleRate(self):
        """Filter sample rate (sps)

        Returns:
            [float] -- sample rate (sps)
        """
        return self._sampRate

    @sampleRate.setter
    def sampleRate(self, newRate):
        self._sampRate = newRate
        self._cutoffFreqNorm = 2 * self._cutoffFreq/self._sampRate
        self._taps = self._designFilter()

    @property
    def cutoffFreq(self):
        """Filter cutoff frequency (Hz)

        Returns:
            [float] -- cutoff frequency (Hz)
        """
        return self._cutoffFreq

    @cutoffFreq.setter
    def cutoffFreq(self, newFreq):
        self._cutoffFreq = newFreq
        self._cutoffFreqNorm = 2 * self._cutoffFreq/self._sampRate
        self._taps = self._designFilter()

    @property
    def windowType(self):
        """Filter window type

        Returns:
            [str] -- filter window type
        """
        return self._windowType

    @windowType.setter
    def windowType(self, newWindow):
        self._windowType = newWindow
        self._taps = self._designFilter()

    # Wrapper around scipy.signal.firwin
    def _designFilter(self):
        """Designs a filter using the window method

        Private method

        Returns:
            [np.array, float] -- FIR filter taps

        """
        return signal.firwin(self._numTaps, self._cutoffFreqNorm, window=self._windowType)

    def process(self, data):
        """Filter samples

        Filters samples using circular convolution in the frequency domain.

        Arguments:
            data {np.array, complex float} -- Complex input samples

        Returns:
            [np.array, complex float] -- Complex filtered samples

        """
        return signal.fftconvolve(data, self._taps, 'full')


class InterpolatingFir(BaseFilter):
    """Interpolate samples with an fir filter
    """
    def __init__(self, sampRate, filtCutoffFreq, newSampRate, windowType="blackman", numTaps=51):
        super(InterpolatingFir, self).__init__(sampRate, filtCutoffFreq, windowType, numTaps)
        self._newSampRate = newSampRate
        self._numInterpSamps = int(newSampRate / sampRate)
        assert self._numInterpSamps > 0, "Interpolation rate must be greater than 0"

    # properties
    @property
    def newSampleRate(self):
        """New output sample rate of the filter

        Returns:
            [float] -- output rate of the filter
        """
        return self._newSampRate

    @newSampleRate.setter
    def newSampleRate(self, newRate):
        if int(newRate / self._sampRate) <= 0:
            warnings.warn("Invalid output rate, ignoring changes", RuntimeWarning)
        else:
            self._newSampRate = newRate
            self._numInterpSamps = int(self._newSampRate / self._sampRate)

    def process(self, data):
        """Interpolates and filters the input data

        Zero pads the input array and then filters with an FIR filter

        Arguments:
            data {complex float} -- Input data

        Returns:
            [complex float] -- Interpolated data
        """
        ret = np.zeros(len(data)*self._numInterpSamps, dtype=data.dtype)
        ret[::self._numInterpSamps] = self._numInterpSamps * data
        return super(InterpolatingFir, self).process(ret)


class DecimatingFir(BaseFilter):
    """Decimates samples with an fir filter
    """
    def __init__(self, sampRate, filtCutoffFreq, newSampRate, windowType="blackman", numTaps=51):
        super(DecimatingFir, self).__init__(sampRate, filtCutoffFreq, windowType, numTaps)
        self._newSampRate = newSampRate
        self._numDeciSamps = int(sampRate / newSampRate)
        assert self._numDeciSamps > 0, "Decimation rate must be greater than 0"

    # properties
    @property
    def newSampleRate(self):
        """New output sample rate of the filter

        Returns:
            [float] -- output rate of the filter
        """
        return self._newSampRate

    @newSampleRate.setter
    def newSampleRate(self, newRate):
        if int(self._sampRate / newRate) <= 0:
            warnings.warn("Invalid output rate, ignoring changes", RuntimeWarning)
        else:
            self._newSampRate = newRate
            self._numDeciSamps = int(self._sampRate / self._newSampRate)

    def process(self, data):
        """Decimates and filters the input data

        Removes samples and then filters with an FIR filter

        Arguments:
            data {complex float} -- Input data

        Returns:
            [complex float] -- Decimated data
        """
        ret = data[::self._numDeciSamps]
        return super(DecimatingFir, self).process(ret)


class ArbResampler():
    """Fir filter based fractional resampler
    """
    def __init__(self, sampRate, filtCutoffFreq, deciRate, interRate, windowType="blackman", numTaps=51):
        self.interp = InterpolatingFir(sampRate, filtCutoffFreq, interRate, windowType, numTaps)
        self.deci = DecimatingFir(sampRate, filtCutoffFreq, deciRate, windowType, numTaps)

    @property
    def sampleRate(self):
        """Filter sample rate (sps)

        Returns:
            [float] -- sample rate (sps)
        """
        if self.interp.sampleRate == self.deci.sampleRate:
            return self.interp.sampleRate
        else:
            raise RuntimeError("Interpolation and decimation sample rates do not match")

    @sampleRate.setter
    def sampleRate(self, newRate):
        self.interp.sampleRate = newRate
        self.deci = newRate

    @property
    def interpolationRate(self):
        """Output sample rate of the interpolator (sps)

        Returns:
            [float] -- sample rate (sps)
        """
        return self.interp.newSampleRate

    @interpolationRate.setter
    def interpolationRate(self, newRate):
        self.interp.newSampleRate = newRate

    @property
    def interpolationFilterCutoff(self):
        """Interpolation filter cutoff frequency

        Returns:
            [float] -- cutoff frequency
        """
        return self.interp.cutoffFreq

    @interpolationFilterCutoff.setter
    def interpolationFilterCutoff(self, newCutoff):
        self.interp.cutoffFreq = newCutoff

    @property
    def decimationFilterCutoff(self):
        """Decimation filter cutoff frequency

        Returns:
            [float] -- cutoff frequency
        """
        return self.deci.cutoffFreq

    @decimationFilterCutoff.setter
    def decimationFilterCutoff(self, newCutoff):
        self.deci.cutoffFreq = newCutoff

    @property
    def cutoffFreq(self):
        return self.interp.cutoffFreq

    @cutoffFreq.setter
    def cutoffFreq(self, newFreq):
        self.interpolationFilterCutoff = newFreq
        self.decimationFilterCutoff = newFreq

    @property
    def decimationRate(self):
        """Output sample rate of the decimator (sps)

        Returns:
            [float] -- sample rate (sps)
        """
        return self.deci.newSampleRate

    @decimationRate.setter
    def decimationRate(self, newRate):
        self.deci.newSampleRate = newRate

    def process(self, data):
        """Applies a simple fractional resampler to the samples

        First interpolates and then decimates the input samples

        Arguments:
            data {complex float} -- Complex input samples

        Returns:
            [complex float] -- resampled data
        """
        return self.deci.process(self.interp.process(data))


class TuneFilterResample(ArbResampler):
    """Frequency shifts, filters, and resamples the input data
    """
    def __init__(self, sampRate, filtCutoffFreq, deciRate, interRate, tunerFreq, blockLength, windowType="blackman", numTaps=51):
        ArbResampler.__init__(self, sampRate, filtCutoffFreq, deciRate, interRate, windowType, numTaps)
        self._tunerFreq = tunerFreq
        self._blockLength = blockLength
        self._sampRate = sampRate
        self._tuner = np.exp(1j * 2*np.pi*self._tunerFreq*np.arange(self._blockLength)/self._sampRate)

    # properties
    @property
    def tunerFrequency(self):
        """Tuner frequency (Hz)

        Returns:
            [float] -- Tuner frequency (Hz)
        """
        return self._tunerFreq

    @tunerFrequency.setter
    def tunerFrequency(self, newFreq):
        self._tunerFreq = newFreq
        self._tuner = np.exp(1j * 2*np.pi*self._tunerFreq*np.arange(self._blockLength)/self._sampRate)

    @property
    def blockLength(self):
        """Number of samples processed

        Returns:
            [float] -- block length of the resampler
        """
        return self._blockLength

    @blockLength.setter
    def blockLength(self, newLength):
        self._blockLength = newLength
        self._tuner = np.exp(1j * 2*np.pi*self._tunerFreq*np.arange(self._blockLength)/self._sampRate)

    @property
    def sampleRate(self):
        """Filter sample rate

        Returns:
            [float] -- sample rate
        """
        return super(TuneFilterResample, self).sampleRate

    @sampleRate.setter
    def sampleRate(self, newRate):
        super(TuneFilterResample, self).sampleRate = newRate
        self._tuner = np.exp(1j * 2*np.pi*self._tunerFreq*np.arange(self._blockLength)/self._sampRate)

    def process(self, data):
        """Frequency shifts, filters, and resamples the input data

        Arguments:
            data {complex float} -- Complex input samples

        Returns:
            [complex float] -- tuned and resampled data
        """
        return ArbResampler.process(self, data[:self._blockLength]*self._tuner)


class Channelizer():
    """Simple FIR filter bank channelizer
    """
    def __init__(self, sampRate, filtCutoffFreq, deciRate, interRate, tunerFreq, blockLength, windowType="blackman", numTaps=51):
        self.filters = []
        for fcut, fd, fi, fc, bl in zip(filtCutoffFreq, deciRate, interRate, tunerFreq, blockLength):
            self.filters.append(TuneFilterResample(sampRate, fcut, fd, fi, fc, bl, windowType, numTaps))

    @property
    def channelCenter(self):
        """Channel center frequency (Hz)

        Arguments:
            channelId {int} -- channel number

        Returns:
            [float] -- center frequency
        """
        return [filter.tunerFrequency for filter in self.filters]

    @channelCenter.setter
    def channelCenter(self, value):
        self.filters[value[0]].tunerFrequency = value[1]
 
    @property
    def sampleRate(self):
        """Sample rate of the filter
        """
        return [filter.sampleRate for filter in self.filters]

    @sampleRate.setter
    def sampleRate(self, value):
        self.filters[value[0]].sampleRate = value[1]

    @property
    def filtCutoff(self):
        """Cutoff  frequency of the filter
        """
        return [filter.filtCutoffFreq for filter in self.filters]

    @filtCutoff.setter
    def filtCutoff(self, value):
        self.filters[value[0]].filtCutoffFreq = value[1]

    @property
    def decimationRate(self):
        """Decimation rate of the filter
        """
        return [filter.decimationRate for filter in self.filters]

    @decimationRate.setter
    def decimationRate(self, value):
        self.filters[value[0]].decimationRate = value[1]

    @property
    def interpolationRate(self):
        """Interpolation rate
        """
        return [filter.interpolationRate for filter in self.filters]

    @interpolationRate.setter
    def interpolationRate(self, value):
        self.filters[value[0]].interpolationRate = value[1]

    @property
    def blockLength(self):
        """Block length of the filter
        """
        return [filter.blockLength for filter in self.filters]

    @blockLength.setter
    def blockLength(self, value):
        self.filters[value[0]].blockLength = value[1]

    def addChannels(self, sampRate, filtCutoffFreq, deciRate, interRate, tunerFreq, blockLength, windowType="blackman", numTaps=51):
        """Add a channel to the filter bank
        
        Arguments:
            sampRate {[type]} -- [description]
            filtCutoffFreq {[type]} -- [description]
            deciRate {[type]} -- [description]
            interRate {[type]} -- [description]
            tunerFreq {[type]} -- [description]
            blockLength {[type]} -- [description]
        
        Keyword Arguments:
            windowType {[type]} -- [description] (default: {"blackman"})
            numTaps {[type]} -- [description] (default: {51})
        """
        for fcut, fd, fi, fc, bl in zip(filtCutoffFreq, deciRate, interRate, tunerFreq, blockLength):
            self.filters.append(TuneFilterResample(sampRate, fcut, fd, fi, fc, bl, windowType, numTaps))

    def process(self, activeChannels, data):
        """Channelize the data

        Arguments:
            activeChannels {iterator} -- Channels to get
            data {complex float} -- Complex float input

        Returns:
            [tuple] -- channel id with base band samples
        """
        return [(channel, self.filters[channel].process(data)) for channel in activeChannels]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sampRateT = 2e6
    cutoffT = 500e3
    filt = Channelizer(sampRateT, [cutoffT], [sampRateT / 2], [2 * sampRateT], [-100e3], [1000])
    filt.channelCenter = (0, 0)
    print(filt.channelCenter)

    samples = np.exp(1j * 2 * 100e3 * np.pi * np.arange(1000) / sampRateT)
    chans = filt.process([0], samples)

    for _, filtSamples in chans:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.abs(samples))
        ax[0].plot(np.abs(filtSamples))

        ax[1].plot(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, 1024)))))
        ax[1].plot(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(filtSamples, 1024)))))
        plt.show()
