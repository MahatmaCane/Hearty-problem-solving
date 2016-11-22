import numpy as np

class TotalActivity:

    """Records number of activated cells at each time step."""

    def __init__(self):
        self.activity = []
        self.time = []

    def record(self, time, activity):
        self.time.append(time)
        self.activity.append(activity)

class Electrogram:

    """Records number of activated cells at each time step."""

    def __init__(self):
        self.activity = []
        self.time = []

    def record(self, time, activity):
        self.time.append(time)
        self.activity.append(activity)

class Ablater:

    """Ablates specified region of myocardium."""

    def __init__(self, one_corner):
        self.first_corner = one_corner

    def set_final_corner(self, corner):
        self.final_corner = corner

class TimeTracker:

    """Steps time forwards & controls termination of simulation when
       animating."""

    def __init__(self, tinit=0, tmax=np.inf):

        self.tmax = tmax
        self.tinit = tinit

    def __iter__(self):

        t = self.tinit
        while t < self.tmax:
            yield t
            if self.stop is True:
                raise StopIteration()
            t += 1

    def stop(self):

        self.stop = True
