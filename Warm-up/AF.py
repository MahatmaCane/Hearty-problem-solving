#### Copyright Joel Dyer, 29/07/2016 ####

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle

from AFModel import Myocardium

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

    def __init__(self, tmax=np.inf):

        self.tmax = tmax
        self.tinit = 0

    def __iter__(self):

        t = 1
        while t < self.tmax:
            yield t
            if self.stop is True:
                raise StopIteration()
            t += 1

    def stop(self):

        self.stop = True

def run(tmax=np.inf, heart_rate=250, tissue_shape=(200, 200), nu=0.8, d=0.05,
        e=0.95, refractory_period=50, animate=True, electrogram=False):

    args = {'heart rate':heart_rate, 'nu':nu, 'delta':d,
            'epsilon':e, 'tmax':tmax}

    myocardium = Myocardium(tissue_shape, nu, d, e, refractory_period)
    tt = TimeTracker(tmax)

    if animate == True:
        ax = plt.gca()
        fig = ax.get_figure()
        qm = ax.pcolorfast(myocardium.counts_until_relaxed, cmap='Greys_r')
        qm.set_array(myocardium.counts_until_relaxed)
        myocardium.determine_number_of_active_cells()
        ax.set_title('0, {}'.format(myocardium._num_active_cells))
        plt.draw()
        plt.pause(0.01)

        def handle_close(evt):
            tt.stop()

        def store_corner(evt):
            global ablater
            ablater = Ablater((evt.xdata, evt.ydata))

        def ablate(evt):
            ablater.set_final_corner((evt.xdata, evt.ydata))
            s.render_unexcitable(ablater.first_corner, ablater.final_corner)

        fig.canvas.mpl_connect('close_event', handle_close)
        fig.canvas.mpl_connect('button_press_event', store_corner)
        fig.canvas.mpl_connect('button_release_event', ablate)
    else:
        print "Beginning simulation with parameters: {}.".format(args)

    if electrogram == True:
        egram = Electrogram()
        myocardium.determine_number_of_active_cells()
        egram.record(0, myocardium._num_active_cells)

    for time in tt:
        myocardium.evolve_wavefront()
        myocardium.update_counts_until_relaxed()
        myocardium.determine_number_of_active_cells()
        if time%heart_rate == 0:
            myocardium.pulse()

        if animate == True:
            qm.set_array(myocardium.counts_until_relaxed)
            ax.set_title('{0}, {1}'.format(time, myocardium._num_active_cells))
            plt.draw()
            plt.pause(0.01)

        if electrogram == True:
            egram.record(time, myocardium._num_active_cells)

    if electrogram == True:
        dirname = 'Data-{0}-{1}'.format(d, e)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(dirname + '/Run-{0}'.format(nu),'w') as fh:
            pickle.dump(np.array([egram.time, egram.activity]), fh)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--shape', '-s', type=int, default=200,
                        help='Myocardium dimensions.')
    parser.add_argument('--nu', '-n', type=float, default=0.21,
                        help='Fraction of existing lateral couplings.')
    parser.add_argument('--delta', '-d', type=float, default=0.05,
                        help='Fraction of defective cells.')
    parser.add_argument('--epsilon', '-e', type=float, default=0.95,
                        help='Probability that defective cell fails to fire.')
    parser.add_argument('--animate', '-a', action='store_true', 
                        help='Animation switch for live animation.')
    parser.add_argument('--electrogram', '-g', action='store_true',
                        help='Switch for recording activity with time.')
    args = parser.parse_args()

    run(nu=args.nu, d=args.delta, e=args.epsilon, animate=args.animate,
        tissue_shape=(200, args.shape), electrogram=args.electrogram)
