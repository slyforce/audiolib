import dataclasses
import functools
import gc
import os
import pprint
import wave
import time
import sys
from typing import Callable

import pyaudio
import numpy as np
import matplotlib.pyplot as plt

from dsp.conversions import wav_signal_to_float, float_signal_to_wav, Converter, ConverterList, PolynomialConverter, \
    RescaleClipping, ChangeVolume, FrequencyBooster, MultiFunctionConverter
from dsp.profiler import Profiler
print(os.getpid())
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paFloat32  # Format of audio samples

converter = ConverterList([
    # PolynomialConverter(1.1),
    # PolynomialConverter(1.2),
    # RescaleClipping(0.5, False),
    ChangeVolume(2.0),

    RescaleClipping(0.8, do_rescale=False),

    #FrequencyBooster(0, 200, 2.0),
    # FrequencyBooster(200, 800, 1.0),
    #FrequencyBooster(1600, 20000, 2.0),

    # RescaleClipping(0.01, do_rescale=False),
])

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    # if "Analogue 1 + 2 (Focusrite USB Audio)" in device_info["name"]:

    if "Focusrite" not in device_info["name"]:
        continue
    if device_info["maxInputChannels"] > 0:
        print(f"Input device {i}: {device_info}")
    elif device_info["maxOutputChannels"] > 0:
        print(f"Output device {i}: {device_info}")


input_device_idx = 22
output_device_idx = 20

def blocking_io_execution():
    p = pyaudio.PyAudio()
    stream_in = p.open(format=FORMAT,
                       channels=1,
                       rate=int(p.get_device_info_by_index(input_device_idx)["defaultSampleRate"]),
                       input=True, # CHUNK,
                       input_device_index=input_device_idx,
                       frames_per_buffer=CHUNK,
                       )  # Replace 1 with the desired device index

    stream_out = p.open(format=FORMAT,
                        channels=1,
                        rate=int(p.get_device_info_by_index(output_device_idx)["defaultSampleRate"]),
                        output=True,
                        output_device_index=output_device_idx,
                        frames_per_buffer=CHUNK,
                        ) # CHUNK)  # Replace 1 with the desired device index

    iters = 0

    iter_since_last_spike = 0
    while True:
        tstart = time.perf_counter_ns()
        data = stream_in.read(1024)
        data = np.frombuffer(data, np.int16)

        x_in = data # wav_signal_to_float(data)
        x_out = converter(x_in)
        #x_out = float_signal_to_wav(x_out)
        #side_by_side_plot(x_in, x_out, "")
        print(x_in.min(), x_in.max(), x_in.std())
        stream_out.write(x_out.tobytes(), 1024)
        #print(x_out.shape)
        lat_ms = (time.perf_counter_ns() - tstart) / 1e6

        iters += 1
        if iters % 200 == 0 or lat_ms > 10:
            print(f"Latency: {lat_ms}ms")

        if lat_ms > 10:
            print(f"high latency after {iter_since_last_spike}, pid={os.getpid()}")
            iter_since_last_spike = 0

        else:
            iter_since_last_spike += 1


def nonblocking_io_execution():


    def callback(in_data, frame_count, time_info, status, converter: ConverterList):
        tstart = time.perf_counter_ns()
        x_in = np.frombuffer(in_data, np.float32)
        x_in = np.reshape(x_in, (-1, 2))
        x_out = converter(x_in)
        print(f"Took {(time.perf_counter_ns() - tstart) / 1e6}ms")
        return x_out.tobytes(), pyaudio.paContinue

    p = pyaudio.PyAudio()
    pprint.pprint(p.get_default_input_device_info())
    pprint.pprint(p.get_default_output_device_info())
    pprint.pprint(p.get_default_host_api_info())

    input_device_idx, output_device_idx = 22, 20
    stream = p.open(
        format=FORMAT,
        channels=2,
        rate=int(p.get_device_info_by_index(input_device_idx)["defaultSampleRate"]),
        input=True,
        input_device_index=input_device_idx,
        output=True,
        output_device_index=20,
        frames_per_buffer=32,
        stream_callback=functools.partial(callback, converter=converter),
        start=True,
    )
    while True:
        time.sleep(1)


if __name__ == "__main__":
    nonblocking_io_execution()
