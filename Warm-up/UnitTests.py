# UnitTests.py

import DataProc as DP
import pickle

params = dict(realisations = 12, tmax = 10e4, heart_rate = 250, 
	    tissue_shape = (200,200), nu = 0.25, d = 0.05, e = 0.05, 
	    refractory_period = 50)

def test_fibrillating(nu, i):
    """Unit test for is_fibrillating() function. Prints the mean_time_in_Af for a given value of nu
       and a given realisation, i, and then plots the activity in order to check it makes sense."""

    # Simulations which prove that the function works with (nu,i): (0.0,0), (0.05,0), (0.1,0), (0.15, 0), (0.15,6), (0.16,1), (0.2,0)
    dirname = 'Data-{0}-{1}-Risk_Curve'.format(params['d'], params['e'])
    activity = pickle.load(open(dirname + '/Run-{0}-{1}-{2}'.format(params['tmax'], nu, i), "r"))
    time_in_AF = DP.is_fibrillating(activity)
    print time_in_AF
    DP.plot_activity(nu,i)

if __name__ == "__main__":

	test_fibrillating(0.15,6)
