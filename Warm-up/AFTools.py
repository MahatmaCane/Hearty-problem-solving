class TotalActivity:

    """Records number of activated cells at each time step."""

    def __init__(self):
        self.activity = []
        self.time = []

    def record(self, time, activity):
        self.time.append(time)
        self.activity.append(activity)