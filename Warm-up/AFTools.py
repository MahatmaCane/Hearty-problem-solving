import numpy as np

class TotalActivity:

    """Records number of activated cells at each time step."""

    def __init__(self):
        self.activity = []

    def record(self, activity):
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

class StatePickler:

    """Object for pickling state of the myocardium."""

    def pickle_state(self, out_dir, myocardium, random_state, t):

        """Record the state of the myocardium, the , and the time step
           in the simulation.

           Input:
            out_dir:        directory in which pickle file is to be stored.
            myocardium:     AFModel.Myocardium instance.
            random_state:   state of numpy RNG as obtained via np.get_state().
            t:              elapsed time in simulation.

           Name of file, to be saved in out_dir, will be 'State-<nu>-<time>'
           where <nu> is the fraction of possible lateral couplings present
           and <time> = t."""

        info = {'myo':myocardium, 'rand_state':random_state, 'time':t}
        with open(out_dir + '/State-{0}-{1}'.format(myocardium._nu, t), 'w') as fh:
            pickle.dump(info, fh)
