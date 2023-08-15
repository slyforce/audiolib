import dataclasses
import functools
import wave
import time
import sys
from typing import Callable

import pyaudio
import numpy as np
import matplotlib.pyplot as plt

from dsp.conversions import wav_signal_to_float, float_signal_to_wav, Converter, ConverterList, PolynomialConverter, \
    RescaleClipping, ChangeVolume, FrequencyBooster, MultiFunctionConverter


def side_by_side_plot(signal1, signal2, destination: str):
    # Load your audio signals (replace these with your actual data loading code)
    # For this example, we'll create two simple sine waves.
    sample_rate = 44100
    duration = 3  # seconds

    plt.clf()
    # Create a figure with two subplots
    N = signal1.shape[0]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot the first signal
    axes[0].plot(list(range(N)), signal1, color='blue')
    axes[0].set_title('Before')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    # Plot the second signal
    axes[1].plot(list(range(N)), signal2, color='green')
    axes[1].set_title('After')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')

    # Adjust layout and show the plots
    plt.tight_layout()

    plt.savefig(destination)
    plt.close(fig)


@dataclasses.dataclass
class FileProcessor:
    wav_file: str
    callback_fn: Callable

    def apply(self):
        with wave.open(wav_filename, 'rb') as wf:
            # Instantiate PyAudio and initialize PortAudio system resources (2)
            p = pyaudio.PyAudio()

            # Open stream using callback (3)
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            input=True,
                            output=True,
                            stream_callback=functools.partial(self.callback_fn, stream=wf))

            # Wait for stream to finish (4)
            while stream.is_active():
                print(stream.get_input_latency(), stream.get_output_latency())
                time.sleep(0.1)

            # Close the stream (5)
            stream.close()

            # Release PortAudio system resources (6)
            p.terminate()



def callback(in_data, frame_count, time_info, status, stream, converter: ConverterList):
    data = np.frombuffer(stream.readframes(frame_count), np.int16)
    x_in = wav_signal_to_float(data)
    x_out = converter(x_in)
    #print(frame_count)
    #side_by_side_plot(x_in, x_out, f"resources/sample1.{frame_count}.png")
    return float_signal_to_wav(x_out), pyaudio.paContinue

wav_filename = "resources/sample1.wav"
converter = ConverterList([
    #PolynomialConverter(1.1),
    #PolynomialConverter(1.2),
    #RescaleClipping(0.05, False),
    #ChangeVolume(15.0),

    #RescaleClipping(0.01, do_rescale=False),

    #FrequencyBooster(0, 200, 10.0),
    #FrequencyBooster(200, 800, 1.0),
    #FrequencyBooster(800, 20000, 10.0),

    #RescaleClipping(0.01, do_rescale=False),

    MultiFunctionConverter([(-np.inf, 0, lambda x: x),
                            (0, 50, lambda x: 200),
                            (50, np.inf, lambda x: x)]),
])


processor = FileProcessor(wav_filename, functools.partial(callback, converter=converter))
processor.apply()
