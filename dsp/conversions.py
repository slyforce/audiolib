import dataclasses
from collections import Callable

import numpy as np
from scipy.signal import butter, lfilter


class Converter:
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplemented


@dataclasses.dataclass
class ConverterList:
    converters: list[Converter]
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        for c in self.converters:
            xs = c(xs)
        return xs


@dataclasses.dataclass
class PolynomialConverter(Converter):
    p: float
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        return xs ** self.p


@dataclasses.dataclass
class RescaleClipping(Converter):
    threshold: float
    do_rescale: bool

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        max_before = np.abs(xs).max()
        xs = np.clip(xs, -self.threshold, self.threshold)
        if self.do_rescale:
            xs = xs / max_before
        return xs


@dataclasses.dataclass
class MultiFunctionConverter:
    interval_to_function: list[tuple[float, float, Callable]]
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(xs)
        for start, end, func in self.interval_to_function:
            out += func(xs) * ((start <= xs) & (xs < end)).astype(np.float32)
        return out



@dataclasses.dataclass
class ChangeVolume(Converter):
    factor: float
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        return xs * self.factor

@dataclasses.dataclass
class FrequencyBooster(Converter):
    start: float
    end: float
    boost: float
    sample_rate: float = 44100

    def _internal_call(self, xs: np.ndarray) -> np.ndarray:
        # Compute the FFT of the audio signal
        fft_audio = np.fft.fft(xs)

        # Find the indices corresponding to the desired frequency range
        frequency_bins = np.fft.fftfreq(len(fft_audio), 1 / self.sample_rate)
        start_index = np.argmin(np.abs(frequency_bins - self.start))
        end_index = np.argmin(np.abs(frequency_bins - self.end))

        # Boost the magnitudes within the specified frequency range
        fft_audio[start_index:end_index + 1] *= self.boost
        # Compute the inverse FFT to get the boosted audio signal
        boosted_audio = np.real(np.fft.ifft(fft_audio))
        return boosted_audio

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        if len(xs.shape) == 2:
            for i in range(xs.shape[1]):
                xs[:,i] = self._internal_call(xs[:,i])
            return xs
        else:
            return self._internal_call(xs)



def wav_signal_to_float(xs: np.ndarray):
    return xs.astype(np.float32) / 2**15


def float_signal_to_wav(xs: np.ndarray):
    return (xs * 2**15).astype(np.int16)

