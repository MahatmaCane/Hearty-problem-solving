#### Copyright Joel Dyer, 29/07/2016 ####

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import time

from AFTools import (Ablater, Loader, StatePickler, TotalActivity, TimeTracker)
from AFModel import Myocardium


def run(tmax=1e3, heart_rate=220, tissue_shape=(200, 200), nu=0.8, d=0.05,
        e=0.05, ref_period=50, animate=True, out_dir=False, plot_egram=False,
        pickle_frequency=None, state_file=None):

    """Run simulation.
    
    Input:
        - tmax:             Total number of time steps. Integer.
        - heart_rate:       Beat period. Integer.
        - tissue_shape:     Dimensions of 2D array. Tuple.
        - nu:               Fraction of possible lateral couplings present. 
                            Float <= 1.
        - d:                Fraction of cells which are defective. Float <= 1.
        - e:                Probability that defective cell doesn't excite given
                            stimulus.
        - ref_period:       Refractory period of cells. Integer.
        - animate:          Switch for animation. Bool.
        - out_dir:          Output directory for data-dumping. If False, no
                            data-dumping occurs.
        - plot_egram:       Switch for plotting activity vs. time at the end of
                            simulation. Bool.
        - pickle_frequency: Integer. Pickle myocardium and state of numpy random
                            number generator after every pickle_frequency time steps.
        - state_file:       Path to file containing desired initial state of the 
                            myocardium and numpy RNG. String."""

    args = {'heart rate':heart_rate, 'epsilon':e, 'delta':d, 'tmax':tmax, 
            'tau':ref_period, 'nu':nu, 'shape':tissue_shape}

    if state_file is not None:
        state_dict = StateLoader(state_file)
        myocardium = state_dict['myo']
        np.random.set_state(state_dict['rand_state'])
        tt = TimeTracker(tinit=state_dict['time'], tmax=tmax)
    else:
        myocardium = Myocardium(tissue_shape, nu, d, e, ref_period)
        tt = TimeTracker(tmax=tmax)

    if animate == True:
        ax = plt.gca()
        fig = ax.get_figure()
        myo_state = myocardium.counts_until_relaxed
        qm = ax.pcolorfast(myo_state, cmap='Greys_r', vmin=0, vmax=ref_period)
        qm.set_array(myocardium.counts_until_relaxed)
        plt.draw()
        plt.pause(0.0001)
        
        textstr = 'Tissue Shape$={0}$ \n'.format(tissue_shape) + r'$\nu$' 
        textstr += '$={0}$ \n$\delta={1}$'.format(nu, d)
        textstr += '\nRefractory Period$={0}$'.format(ref_period)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax.text(0.75, 0.15, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        def handle_close(evt):
            tt.stop()

        def store_corner(evt):
            global ablater
            ablater = Ablater((evt.xdata, evt.ydata))

        def ablate(evt):
            ablater.set_final_corner((evt.xdata, evt.ydata))
            myocardium.render_unexcitable(ablater.first_corner,
                                          ablater.final_corner)

        fig.canvas.mpl_connect('close_event', handle_close)
        fig.canvas.mpl_connect('button_press_event', store_corner)
        fig.canvas.mpl_connect('button_release_event', ablate)
    else:
        print "Beginning simulation with parameters: {}.".format(args)

    if out_dir is not False:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        egram = TotalActivity()
        egram.record(myocardium.number_of_active_cells())

    for time in tt:
        if out_dir is not False:
            if pickle_frequency is not None:
                if time%pickle_frequency == 0:
                   rand_state = np.random.get_state()
                   StatePickler().pickle_state(out_dir, myocardium,
                                               rand_state, time)

        if time%heart_rate == 0:
            myocardium.evolve(pulse=True)
        else:
            myocardium.evolve()

        if animate == True:
            qm.set_array(myocardium.counts_until_relaxed)
            count = myocardium.number_of_active_cells()
            ax.set_title('Time: {0}, Active Cells: {1}'.format(time, count))
            plt.draw()
            plt.pause(0.0001)

        if out_dir is not False:
            egram.record(myocardium.number_of_active_cells())

    if out_dir is not False:
        with open(out_dir + '/Run-{0}'.format(nu),'w') as fh:
            pickle.dump(np.array([egram.activity]), fh)
            
    if plot_egram == True:
        fig = plt.figure(2)
        plt.scatter(egram.activity)
        plt.title('Total Cell Activity')
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--shape', '-s', type=int, default=200,
                        help='Myocardium dimensions.')
    parser.add_argument('--nu', '-n', type=float, default=0.21,
                        help='Fraction of existing lateral couplings.')
    parser.add_argument('--delta', '-d', type=float, default=0.05,
                        help='Fraction of defective cells.')
    parser.add_argument('--epsilon', '-e', type=float, default=0.05,
                        help='Probability that defective cell fails to fire.')
    parser.add_argument('--animate', '-a', action='store_true', 
                        help='Animation switch for live animation.')
    parser.add_argument('--out_dir', '-o', type=str, default=False,
                        help='Output directory for data dumping.')
    parser.add_argument('--plot_egram', '-p', action='store_true',
                        help="""Switch for plotting electrogram activity\n 
                              against time.""")
    parser.add_argument('--pickle_frequency', '-f', type=int, default=None,
                        help="""Inverse frequency at which to pickle state of\n
                              the simulations. Requires a specified out_dir.""")
    parser.add_argument('--state_file', '-m', type=str, default=None,
                        help='Pickle file containing pickled myocardium.')
    args = parser.parse_args()

    run(nu=args.nu, d=args.delta, e=args.epsilon, animate=args.animate,
        tissue_shape=(200, args.shape), out_dir=args.out_dir,
        pickle_frequency=args.pickle_frequency, state_file=args.state_file)
