import datetime
import time


class Timer:
    def __init__(self):
        self.prev_time = None

    def __enter__(self):
        self.prev_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_passed = time.time() - self.prev_time
        print('Time passed:', datetime.timedelta(seconds=int(time_passed)))
