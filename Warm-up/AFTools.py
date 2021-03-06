import glob
import os
import numpy as np
import pickle

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

def prepare_axes(ax, title=None, xlabel=None, ylabel=None):

    ax.grid()
    if title is not None:
        ax.set_title(title)
    if not (xlabel is None):
        ax.set_xlabel(xlabel)
    if not (ylabel is None):
        ax.set_ylabel(ylabel)
    return ax

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
        with open(out_dir + '/State-{0}'.format(t), 'w') as fh:
            pickle.dump(info, fh)

    def sequence_to_csv(self, sequence, location):

        with open(location, 'w') as fh:
            for i in sequence:
                fh.write('{0}\n'.format(i))


class Loader:

    def load_state(self, path_to_file):

        with open(path_to_file, 'r') as fh:
            self.state = pickle.load(fh)
        return self.state

def get_nus(directory):

    """Input: patient or directory? I think directory."""
    nus = set()
    for fname in glob.glob(directory + "/sim-*"):
        nu = fname.split("-")[8]
        nus.add(float(nu))

    return sorted(nus)

def no_reals(dirname, nus):

    no_reals = []
    for nu in nus:
        files = dirname + "/Run-*-nu-{0}".format(nu)
        reals = len(glob.glob(files))
        no_reals.append(reals)
    return dict(zip(nus, no_reals))
