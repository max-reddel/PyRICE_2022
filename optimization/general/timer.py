"""
This module introduces a quick util object of time.
"""

import time


class Timer:
    """
    A timer object to track time of computations.
    """

    def __init__(self, tracking=True):
        """
        @param tracking: Boolean: whether to keep track and print the duration or not.
        """

        self.tracking = tracking
        self.start_time = time.time()

    def stop(self):
        """
        Stop timer and print formatted duration.
        """
        if self.tracking:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(self.duration))
            print(f"Runtime: {formatted_time}")
