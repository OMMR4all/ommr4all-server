from timeit import default_timer as timer
from logging import getLogger
from itertools import tee

logger = getLogger()


class PerformanceCounter:

    def __init__(self, function_name):
        self.function_name = function_name

    def __enter__(self):
        self.start = timer()
        print("123")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = timer()
        logger.info("Time needed for function {}: {} secs \n".format(self.function_name, end - self.start))
        print("1234")


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
