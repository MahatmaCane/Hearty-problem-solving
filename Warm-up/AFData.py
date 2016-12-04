import glob
import numpy as np
import pickle

from matplotlib import pyplot as plt


class Basin:

    def __init__(self, data, percentile, offset=0):

        """data:       1D array of mean time in configs with activity A.
           percentile: percentage of data to be included when defining the
                       width. Not in normalised form (i.e. max value 100)."""

        self.offset = offset
        self.bounds = self.__get_bounds(data, percentile)

    def __get_bounds(self, data, percentile):

        norm_data = data / np.sum(data)
        effectivePercentile = float(100 - percentile) / (2. * 100)
        forward_cum_dist_func = np.cumsum(norm_data)
        backwards_cum_dist_func = np.cumsum(norm_data[::-1])
        lower_bound = np.argmax(forward_cum_dist_func > effectivePercentile)
        upper_bound = np.argmax(backwards_cum_dist_func > effectivePercentile)
        # Dist function was reversed to find upper_bound location so need to
        # take it away from total number of points to get actual location
        upper_bound = data.size - upper_bound
        bounds = (self.offset + lower_bound, self.offset + upper_bound)

        return bounds

    def __repr__(self):

        return """Basin:
                    Lower  - {0}
                    Upper  - {1}""".format(self.bounds[0], self.bounds[1])

class Averager:

    def __init__(self):

        self.avgd_data = None

    def add_to_average(self, data):

        if self.avgd_data is None:
            self.avgd_data = np.array(data)
        else:
            if self.avgd_data.size < data.size:
                tmp_avgs = data
                tmp_avgs[:self.avgd_data.size] += self.avgd_data
                self.avgd_data = tmp_avgs
            else:
                self.avgd_data[:data.size] += data

    def normalise(self, norm_const):

        self.avgd_data /= norm_const


def __prepare_axes(ax, title=None, xlabel=None, ylabel=None):

    ax.grid()
    if title is not None:
        ax.set_title(title)
    if not (xlabel is None):
        ax.set_xlabel(xlabel)
    if not (ylabel is None):
        ax.set_ylabel(ylabel)
    return ax


def find_time_at_activity(data):

    time_act = data
    if isinstance(time_act, list):
        time_act = np.array(time_act)
    try:
        activity = time_act[0, :]
    except IndexError:
        activity = time_act
    time_at_activity = np.bincount(activity) / float(activity.size)

    return time_at_activity


def basins_same_axes(files, nu, avg=False):

    """Plot multiple realisations at a given nu on the same axis."""

    if avg is False:
        ax = plt.gca()
        ax.clear()
    elif avg is True:
        fig, (ax, avg_ax) = plt.subplots(2, 1, sharex=True)

    num_colours = len(glob.glob(files))
    cmap = plt.get_cmap('rainbow')

    y_label = "$Fraction\ of\ time\ in\ configurations\ with\ activity\ A$"
    ax = __prepare_axes(ax, title=r"$\nu={0}$".format(nu), ylabel=y_label)

    i = 1
    averager = Averager()
    for fname in glob.glob(files):

        with open(fname, 'r') as fh:
            time_act = pickle.load(fh)
            time_at_activity = find_time_at_activity(time_act)
            all_activity_vals = np.arange(np.max(activity)+1)
            ax.plot(all_activity_vals, time_at_activity, '-',
                    c=cmap(1.*i/num_colours))
            if avg is True:
                averager.add_to_average(time_at_activity)

        i += 1

    y_label = "$Mean\ fraction\ of\ time\ in\ config's\ with\ activity\ A$"
    if avg is True:
        averager.normalise(i)
        avg_ax = __prepare_axes(avg_ax, xlabel="$Activity\ A$", ylabel=y_label)
        avg_ax.plot(averager.avgd_data)

    plt.show(block=False)


def avg_and_save(files, filename):

    i = 1
    averager = Averager()
    for fname in glob.glob(files):

        with open(fname, 'r') as fh:
            time_act = pickle.load(fh)
            time_at_activity = find_time_at_activity(time_act)
            averager.add_to_average(time_at_activity)

        i += 1

    averager.normalise(i)

    with open(filename, 'w') as fh:
        pickle.dump(averager.avgd_data, fh)


def determine_basin_boundaries(fname, region, percentile=68.3):

    """Input:
            data:       2D array of mean time in configs with activity A in the
                        vicinity of an attractor, in order of increasing A.
            region:     slice object indicating region of data to consider.
            percentile: percentage of data in input array data to be enclosed 
                        by the basin boundaries which are to be calculated 
                        here.

       Output:
            Basin instance (see above)."""

    with open(fname, 'r') as fh:
        data = pickle.load(fh)

    selected_data = data[region]
    basin = Basin(selected_data, percentile, offset=region.start)

    print basin
    return basin


def prob_enter_basin(path_to_file, basin):

    """Determine probability that myocardium enters fibrillatory attractor
       given an instantaneous activity x.
       
       Input:
            - path_to_file, """

    with open(path_to_file, 'r') as fh:
        time_act = pickle.load(fh)


def plot_activity(files):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = __prepare_axes(ax)

    for fname in glob.glob(files):
        with open(fname, 'r') as fh:
            time_act = pickle.load(fh)
            if isinstance(time_act, list):
                time_act = np.array(time_act)
            try:
                activity = time_act[0, :]
            except IndexError:
                activity = time_act

            ax.plot(activity)
        
    plt.show(block=False)


def plot_next_activity(fname):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = __prepare_axes(ax) 

    with open(fname, 'r') as fh:
        time_act = pickle.load(fh)
        if isinstance(time_act, list):
            time_act = np.array(time_act)
        try:
            activity = time_act[0, :]
        except IndexError:
            activity = time_act

        ax.scatter(activity[:-1], activity[1:])

    plt.show(block=False)
