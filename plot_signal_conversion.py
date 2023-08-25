import bisect

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter

from dsp.conversions import wav_signal_to_float, float_signal_to_wav, Converter, ConverterList, PolynomialConverter, \
    RescaleClipping, ChangeVolume, FrequencyBooster, MultiFunctionConverter, DigitalFilter, create_butterworth_lowpass, \
    create_butterworth_highpass


def side_by_side_plot(signal1: np.ndarray, signal2: np.ndarray, sample_rate: float = 44100):
    # Create a figure with two subplots
    N = signal1.shape[0]
    if len(signal1.shape) == 2:
        assert signal1.shape[1] == 1, "Signal must have only a single-channel"
        signal1 = signal1.reshape((-1, ))

    if len(signal2.shape) == 2:
        assert signal2.shape[1] == 1, "Signal must have only a single-channel"
        signal2 = signal2.reshape((-1, ))

    half_N = int(N // 2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot the first signal
    axes[0].plot(list(range(N)), signal1, color='blue', label="before")
    axes[0].plot(list(range(N)), signal2, color='red', label="after")
    axes[1].set_title('Time spectrum')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    # Plot the second signal
    # Compute the Fourier transform
    frequencies = np.fft.fftfreq(N, 1./sample_rate)[:half_N]
    ft1 = np.abs(np.fft.fft(signal1))[:half_N]
    ft2 = np.abs(np.fft.fft(signal2))[:half_N]

    q1 = bisect.bisect_left(frequencies, 0)
    q2 = bisect.bisect_left(frequencies, 100)

    # Plot the Fourier spectrum
    axes[1].plot(frequencies[q1:q2], ft1[q1:q2], color='blue', label="before")
    axes[1].plot(frequencies[q1:q2], ft2[q1:q2], color='red', label="after")
    axes[1].set_title('Frequency spectrum')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Absolute spectrum ')

    # Adjust layout and show the plots
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    time = np.linspace(0, 1, 44100)  # Time values
    s1 = np.sin(2 * np.pi * 20 * time)
    s1 += 0.5 * np.sin(2 * np.pi * 8 * time)
    s1 = np.reshape(s1, (s1.size, 1))
    
    converters = ConverterList([
        #ChangeVolume(2.0),
        #RescaleClipping(0.8, False),
        #FrequencyBooster(1, 6, 20.0),
        #DigitalFilter(np.array([0.5]), np.array([0.5, 1.0]), 1),
        #create_butterworth_lowpass(4, 3, fs=44100),
        create_butterworth_highpass(4, 20, fs=44100),

    ])

    s2 = converters(s1)
    side_by_side_plot(s1, s2, sample_rate=44100)


if __name__ == "__main__":
    main()
