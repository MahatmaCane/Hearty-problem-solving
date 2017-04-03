import glob
import numpy as np

import get_no_reals as gnr

from AFTools import StatePickler
from matplotlib import pyplot as plt
import AFDataProc as AFDP

plt.ion()

kish_nus = [0.02, 0.04, 0.06, 0.08, 0.11,
            0.13, 0.15, 0.17, 0.19, 0.21,
            0.23, 0.25, 0.27, 0.29, 0.12,
            0.14, 0.16, 0.18, 0.2, 0.22,
            0.24, 0.26, 0.28, 0.3, 0.1]

kish_mean = [0.99981, 0.99983,0.9998,0.99968, 0.99772, 0.96099,
             0.60984, 0.16381, 0.017807, 0.020737, 4.922e-05, 0.0001084,
             0,0, 0.99152, 0.86184, 0.29714, 0.039206, 0.0056277,
             4.834e-05, 0.00082172, 0,0,9.406e-05, 0.99919]

kish_err = [4.3015e-06, 3.8088e-06, 1.0454e-05, 3.0663e-05, 0.00044859,
            0.018246, 0.054379, 0.041092, 0.0080603, 0.016513, 4.8685e-05,
            8.4968e-05, 0, 0, 0.0027053, 0.028043, 0.055185, 0.013863,
            0.0028284, 3.6005e-05, 0.00081342, 0, 0, 9.3115e-05, 0.00010423]

def time_fibrillating(series, threshold=220, T=220):

    """T - beat period"""

    series = np.array(series)
    above_threshold = (series > 210).astype(np.int)
    flip_switch = above_threshold[1:] - above_threshold[:-1]
    # Times at which system goes above threshold from below
    exceed_thresh = set(*np.where(flip_switch == 1))
    # Times at which system goes below threshold from above
    below_thresh = set(*np.where(flip_switch == -1))

    # If no elements in below_thresh, system never returns to sinus rhythm
    if len(below_thresh) == 0:
        mean_time_fib = np.sum(above_threshold)/float(above_threshold.size)
        return mean_time_fib

    below = False
    # If time of exceeding threshold from below is before first time threshold
    # crossed from above, system was initially below. This should always
    # evaluate to True under model conditions studied so far.
    if min(exceed_thresh) < min(below_thresh):
        below = True
        exceed_thresh.remove(min(exceed_thresh))

    while len(exceed_thresh) > 0:
        if len(below_thresh) == 0:
            break
        min_to_below = min(below_thresh)
        min_to_above = min(exceed_thresh)
        time_to_next_beat = (int(min_to_below/float(T)) + 1)*T - min_to_below
        # If time to next time we exceed threshold contains two or more
        # heart beats, then it's exited fibrillation. Else, still fib.
        if min_to_above - min_to_below < T + time_to_next_beat:
            # +1 to indeces because flip_switch has elements indexed 1-size
            # of time series
            above_threshold[min_to_below+1:min_to_above+1] = 1
        below_thresh.remove(min_to_below)
        exceed_thresh.remove(min_to_above)

    mean_time_fib = np.sum(above_threshold)/float(above_threshold.size)
    print mean_time_fib
    return mean_time_fib

def plot_risk_curve(dirname, nus=kish_nus, delta=0.05, epsilon=0.05, L=200, tau=50):

    try:
        means_stds = np.genfromtxt(dirname + '/plot_data', delimiter=",")
        mean_times_in_AF = means_stds[:,0]
        std_devs = means_stds[:,1]
    except IOError:
        mean_times_in_AF = []
        std_devs = []
        reals = gnr.no_reals(dirname, nus)

        # Get mean time in AF for each nu
        for nu in nus:
            real = reals[nu]
            time_in_AF = []

            # Get mean time in AF for each realisation at given nu
            print "NU = {0}, REALISATIONS = {1}".format(nu, real)
            for fname in glob.glob(dirname + "/Run-*-nu-{0}".format(nu)):
                activity = np.genfromtxt(fname)
                # Swambo's code
                # mean_time_in_fib = AFDP.mean_time_fibrillating(activity)
                mean_time_in_fib = time_fibrillating(activity)
                time_in_AF.append(mean_time_in_fib)

            mean_times_in_AF.append(np.mean(time_in_AF))
            std_dev = np.std(time_in_AF)/(real)**(0.5)
            std_devs.append(std_dev)

            print nu, time_in_AF, std_dev
        # Save data
        with open(dirname + "/plot_data", 'w') as fh:
            for mt, std in zip(mean_times_in_AF, std_devs):
                print "Saving {0}, {1}".format(mt, std)
                fh.write("{0}, {1}\n".format(mt, std))

    fig, (ax) = plt.subplots(1, 1)
    np_nus = np.arange(0, max(nus), 0.01)
    anal = 1 - (1 - (1 - np_nus)**tau)**(delta * L**2)
    ax.plot(np_nus, anal, '--', label='Analytic')
    ax.errorbar(nus, mean_times_in_AF, yerr=std_devs, fmt='x', label="Our data")
    ax.errorbar(nus, kish_mean, yerr=kish_err, fmt='x', label="Kishan's data")
    ax.set_xlabel(r"$\nu$", fontsize=30)
    ax.set_ylabel("Mean time in/Risk of AF", fontsize=30)
    ax.grid()
    plt.legend()
    plt.show()
