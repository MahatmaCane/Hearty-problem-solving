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

def run(tmax=1e5, heart_rate=220, tissue_shape=(200, 200), nu=0.08, d=0.05,
        e=0.05, ref_period=50, animate=True, dump_loc=None, pickle_period=None,
        state_file=None, new_RNG=True):

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
        - dump_loc:         Output directory for data-dumping. If False, no
                            data-dumping occurs.
        - pickle_period:    Integer. Pickle myocardium and state of numpy random
                            number generator after every pickle_period time steps.
        - state_file:       Path to file containing desired initial state of the 
                            myocardium and numpy RNG. String.
        - new_RNG:          Bool. Switch for using a new state for loaded numpy
                            or using pickled state."""

    args = {'heart rate':heart_rate, 'epsilon':e, 'delta':d, 'tmax':tmax, 
            'tau':ref_period, 'nu':nu, 'shape':tissue_shape,
            'state_file':state_file, 'new_RNG':new_RNG}

    if pickle_period is not None:
        assert dump_loc is not None, "Output directory required."

    if dump_loc is not None:
        out_dir = os.path.dirname(dump_loc)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # Initialising myocardium
    if state_file is not None:
        state_dict = Loader().load_state(state_file)
        myocardium = state_dict['myo']
        myocardium.determine_lateral_couplings(nu)
        if new_RNG is False:
            np.random.set_state(state_dict['rand_state'])
        tt = TimeTracker(tinit=state_dict['time'], tmax=tmax)
    else:
        myocardium = Myocardium(tissue_shape, d, e, ref_period)
        rand_state = np.random.get_state()
        if dump_loc is not None:
            StatePickler().pickle_state(out_dir, myocardium, rand_state, 0)
        myocardium.determine_lateral_couplings(nu)
        tt = TimeTracker(tmax=tmax)

    # Animation-related code
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

    # Record activity if location for dumping given. 
    if dump_loc is not None:
        total_activity = TotalActivity()
        total_activity.record(myocardium.number_of_active_cells())

    # Iterating simulation
    for time in tt:
        if dump_loc is not None:
            if pickle_period is not None:
                if time%pickle_period == 0:
                   rand_state = np.random.get_state()
                   StatePickler().pickle_state(out_dir, myocardium,
                                               rand_state, time)

        if time%heart_rate == 0:
            myocardium.evolve(pulse=True)
        else:
            myocardium.evolve()

        # Code for updating animation
        if animate == True:
            qm.set_array(myocardium.counts_until_relaxed)
            count = myocardium.number_of_active_cells()
            ax.set_title('Time: {0}, Active Cells: {1}'.format(time, count))
            plt.draw()
            plt.pause(0.0001)

        # Record activity if out_dir given
        if dump_loc is not None:
            total_activity.record(myocardium.number_of_active_cells())

    # Dump activity list
    if dump_loc is not None:
        StatePickler().sequence_to_csv(total_activity.activity, dump_loc)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--total_time', '-t',  type=float, default=1e5,
                        help='Total simulation time.')
    parser.add_argument('--shape', '-s', type=int, default=200,
                        help='Myocardium dimensions.')
    parser.add_argument('--nu', '-n', type=float, default=0.1,
                        help='Fraction of existing lateral couplings.')
    parser.add_argument('--delta', '-d', type=float, default=0.01,
                        help='Fraction of defective cells.')
    parser.add_argument('--epsilon', '-e', type=float, default=0.05,
                        help='Probability that defective cell fails to fire.')
    parser.add_argument('--animate', '-a', action='store_true', 
                        help='Animation switch for live animation.')
    parser.add_argument('--dump_loc', '-l', type=str, default=None,
                        help='Output location for data dumping.')
    parser.add_argument('--pickle_period', '-f', type=int, default=None,
                        help="""Period at which to pickle state of\n
                              the simulations. Requires a specified dump_loc.""")
    parser.add_argument('--state_file', '-m', type=str, default=None,
                        help='Pickle file containing pickled myocardium.')
    parser.add_argument('--RNG', action='store_true',
                        help="""Bool switch for using pickled numpy RNG state
                                or using new state.""")
    args = parser.parse_args()

    run(tmax=args.total_time, nu=args.nu, d=args.delta, e=args.epsilon, 
        animate=args.animate, tissue_shape=(args.shape, args.shape), 
        dump_loc=args.dump_loc, pickle_period=args.pickle_period, 
        state_file=args.state_file, new_RNG=args.RNG)
