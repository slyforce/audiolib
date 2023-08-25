import dataclasses
from collections import Callable

import numpy as np
from scipy.signal import butter, lfilter


class Converter:
    def __call__(self, xs: np.ndarray) -> np.ndarray:
        raise NotImplemented


class DigitalFilter(Converter):
    x_buffer: np.ndarray
    y_buffer: np.ndarray

    # Should be specified in the order of oldest to most recent!
    # e.g. w_{t-3}, w{t-2}, w{t-1}
    x_coeffs: np.ndarray
    y_coeffs: np.ndarray

    def __init__(self, x_coeffs: np.ndarray, y_coeffs: np.ndarray, num_channels: int):
        self.x_buffer = np.zeros((x_coeffs.size, num_channels))
        self.y_buffer = np.zeros((y_coeffs.size-1, num_channels))
        self.x_coeffs = x_coeffs.reshape((-1, 1))
        self.y_coeffs = y_coeffs.reshape((-1, 1))

    def __call__(self, xs: np.ndarray) -> np.ndarray:
        # todo: work with one array of length seq_len + buffer_len instead of always reallocating one
        xs_buffer = np.concatenate([self.x_buffer, xs], axis=0)
        ys_buffer = np.concatenate([self.y_buffer, xs], axis=0)
        out, self.x_buffer, self.y_buffer = self._apply(xs_buffer,
                                                  ys_buffer, self.x_coeffs, self.y_coeffs)
        return out

    def _apply(self,
               xs: np.ndarray,
               ys: np.ndarray,
               xs_w: np.ndarray,
               ys_w: np.ndarray):
        xs_buf_size = len(xs_w)
        ys_buf_size = len(ys_w) - 1
        start = ys_buf_size
        # todo: further optimise this by calling the convolution operator on the input sequence
        for t in range(start, ys.size):
            x_c = np.dot(xs_w.T, xs[t + 1 - xs_buf_size:t + 1])
            y_c = 0
            if ys_buf_size:
                y_c = np.dot(ys_w[:-1].T, ys[t - ys_buf_size:t])
            ys[t] = (x_c - y_c) / ys_w[-1, :]

        xs_buffer = xs[-xs_buf_size:]
        ys_buffer = ys[-ys_buf_size:]
        ys = ys[ys_buf_size:]
        return ys, xs_buffer, ys_buffer


def create_butterworth_lowpass(order: int, freq: float, fs: float, num_channels: int = 1):
    x_coeffs, y_coeffs = butter(order, freq, fs=fs)
    return DigitalFilter(x_coeffs[::-1], y_coeffs[::-1], num_channels)


def create_butterworth_highpass(order: int, freq: float, fs: float, num_channels: int = 1):
    x_coeffs, y_coeffs = butter(order, freq, btype="high", fs=fs)
    return DigitalFilter(x_coeffs[::-1], y_coeffs[::-1], num_channels)



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

