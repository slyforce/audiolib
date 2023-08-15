import cProfile
import pstats
from contextlib import contextmanager


class Profiler:
    def __init__(self):
        self.output = False

    def __enter__(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.output:
            self.profiler.disable()
            self.profiler.create_stats()
            stats = pstats.Stats(self.profiler)

            stats.strip_dirs().sort_stats("cumulative")
            stats.print_stats()

