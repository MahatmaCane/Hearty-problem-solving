#print activity lists at each nu to see wagwan famalam

import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
from AFModel import Myocardium
from AFTools import TimeTracker, TotalActivity, prepare_axes, StatePickler

params = dict(realisations = 60, tmax = 10e4, heart_rate = 250, 
              tissue_shape = (200,200), nu = 0.25, d = 0.01, e = 0.05, 
              refractory_period = 50)

##### Generic functions used throughout ####

def enters_fib(t, activity, threshold):
    """ Returns time at which the system first enters fibrillation from time t to time tmax,
        unless the system does not enter fibrillation, then an IndexError is raised. """
    for i in range(t,len(activity)):
        if activity[i] >= threshold:
            return i
        if i+1 == len(activity): 
            raise IndexError        # Simulation Terminated in Sinus Rhythm

def leaves_fib(t, activity, threshold):
    """ Returns time at which the system first leaves fibrillation from time t to time tmax,
        unless the system does not leave fibrillation, then an IndexError is raised."""

    if t > params['tmax'] - params['heart_rate']:
        raise ValueError

    heart_beat_times = [j for j in range(0,len(activity)) if j%params['heart_rate'] == 0]
    b = 0

    for i in range(0, len(heart_beat_times)):  #Finds time of first heart beat after entering fibrillation and assigns b that value.
        if heart_beat_times[i] > t:            #Where c is the time that the system entered fibrillation
            b = heart_beat_times[i] 
            break

    test_activities = [i for i in range(b,len(activity), params['heart_rate'])]
    if len(test_activities) <= 2: #Problem arises when the sim terminates in fibrillation but there is not two heartbeats to check.
        raise ValueError
    for i in test_activities[:-1]:
        if activity[i] <= threshold:
            if activity[i+params['heart_rate']] <= threshold:
                # An additional constraint on defining sinus rhythm. The average activity is less than the width of the myocardium
                # since the heartbeat is longer (250 > 200), i.e. there will be a period in sinus rhythm where there is 0 activity.
                if activity[i-1:i-1+params['heart_rate']] == []:
                    print "For i = %i and t = %i, EMPTY SLICE"%(i,t)
                    continue
                if np.mean(activity[i-1:i-1+params['heart_rate']]) <= params['tissue_shape'][0]:  
                    return i
        if i == test_activities[-2]:  #Not sure why it doesn't reach [-1]???
            raise IndexError # Simulation Terminated in Fibrillatory state

def cross_boundary(t, activity, boundary, cross_from = 'below'):
    """ Returns time at which the system first crosses boundary (from below or above) from time 
        t to time tmax, unless the simulation terminates before such a transition, then an 
        IndexError is raised. """
    if cross_from == 'below':
        for i in range(t, len(activity)):
            if activity[i] >= boundary:
                return i
            if i+1 == len(activity):
                raise IndexError       
    
    elif cross_from == 'above':
        for i in range(t, len(activity)):
            if activity[i] <= boundary:
                return i
            if i+1 ==len(activity):
                raise IndexError

################# RISK CURVE ######################

def generate_risk_curve_data():
    """ Generates the total_activity lists for each realisation of each nu, and saves them
        to a folder with appropriate file names. Saves the list of nus used."""

    nus = [0.02, 0.04, 0.06, 0.08, 0.11, 
           0.13, 0.15, 0.17, 0.19, 0.21, 
           0.23, 0.25, 0.27, 0.29, 0.12, 
           0.14, 0.16, 0.18, 0.2, 0.22, 
           0.24, 0.26, 0.28, 0.3, 0.1]

    dirname = 'Risk-Curve-{0}-{1}-{2}'.format(params['d'], params['e'], params['tmax'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for nu in nus:
        time_in_AF = []
        for i in range(0,params['realisations']):

            myocardium = Myocardium(params['tissue_shape'], nu, params['d'], 
                                    params['e'], params['refractory_period'])
            total_activity = TotalActivity()
            tt = TimeTracker(params['tmax'])
            total_activity.record(myocardium.number_of_active_cells())

            for time in tt:
                if time%params['heart_rate'] == 0:
                    myocardium.evolve(pulse=True)
                else:
                    myocardium.evolve()
                total_activity.record(myocardium.number_of_active_cells())


            with open(dirname + '/Run-{0}-nu-{1}'.format(i, nu),'w') as fh:
                pickle.dump(np.array(total_activity.activity), fh)
            # print "Run {0}/{1} saved for nu = {2}.".format(i+1, params['realisations'], nu)

    with open(dirname + '/Nus-{0}'.format(params['tmax']),'w') as fh:
        pickle.dump(np.array(nus), fh)

def mean_time_fibrillating(activity):
    a = 0
    entersfib = []
    leavesfib = []
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]
    
    while a < len(activity):
        try:
            i = enters_fib(a, activity, threshold)
            entersfib.append(i)
            a = i
            j = leaves_fib(a, activity, threshold)
            leavesfib.append(j)
            a = j
        except IndexError:
            break

    #Creating a list of 0's and 1's for each time step to note whether in fibrillatory state or not. This simplifies calculating mean_time_in_AF.
    try:
        is_fibrillating = [0 for i in range(0,entersfib[0])]
    except IndexError:
        return 0  #System never enters fibrillation
    
    x = len(entersfib)
    y = len(leavesfib)

    for k in range(0, max(x,y)):
        try:
            is_fibrillating.extend([1 for i in range(entersfib[k],leavesfib[k])])
            is_fibrillating.extend([0 for j in range(leavesfib[k],entersfib[k])])
        except IndexError:
            if is_fibrillating[-1] == 1:
                is_fibrillating.extend([0 for j in range(leavesfib[k],int(params['tmax']))])
            else:
                is_fibrillating.extend([1 for i in range(entersfib[k],int(params['tmax']))])
            break

    return np.mean(is_fibrillating)

def risk_curve_plot():

    dirname = 'Risk-Curve-{0}-{1}-{2}'.format(params['d'], params['e'], params['tmax'])
    nus = pickle.load(open(dirname + '/Nus-{0}'.format(params['tmax']), "r"))

    mean_time_in_AF = []
    std_devs = []

    for nu in nus:
        time_in_AF = []

        for i in range(0, params['realisations']):
            try:
                activity = pickle.load(open(dirname + '/Run-{0}-nu-{1}'.format(i, nu), "r"))
                # is_fibrillating = np.array(activity) >= (params['tissue_shape'][0]+0.05*params['tissue_shape'][0])
                # time_in_AF.append(float(np.sum(is_fibrillating))/ params['tmax'])
                mean_time_in_fib = mean_time_fibrillating(activity)
                time_in_AF.append(mean_time_in_fib)
            except:
                Exception("Data for realisation %i does not exist"%(i))
        mean_time_in_AF.append(np.mean(time_in_AF))
        std_devs.append(np.std(time_in_AF))

    kishanNus = [0.02, 0.04, 0.06, 0.08, 0.11, 
                 0.13, 0.15, 0.17, 0.19, 0.21, 
                 0.23, 0.25, 0.27, 0.29, 0.12, 
                 0.14, 0.16, 0.18, 0.2, 0.22, 
                 0.24, 0.26, 0.28, 0.3, 0.1]
    
    kishanMeanAF = [0.99981, 0.99983,0.9998,0.99968, 0.99772, 0.96099, 
                    0.60984, 0.16381, 0.017807, 0.020737, 4.922e-05, 0.0001084,
                    0,0, 0.99152, 0.86184, 0.29714, 0.039206, 0.0056277,
                    4.834e-05, 0.00082172, 0,0,9.406e-05, 0.99919]
    
    kishanStdDevs = [4.3015e-06, 3.8088e-06, 1.0454e-05, 3.0663e-05, 0.00044859, 
                     0.018246, 0.054379, 0.041092, 0.0080603, 0.016513, 4.8685e-05, 
                     8.4968e-05, 0, 0, 0.0027053, 0.028043, 0.055185, 0.013863,
                     0.0028284, 3.6005e-05, 0.00081342, 0, 0, 9.3115e-05, 0.00010423]

    plt.errorbar(kishanNus,kishanMeanAF, yerr = kishanStdDevs, xerr=None, fmt='x', label = 'Data from Kishan')

    analytic = [(1 - (1 - (1 - nu)**params['refractory_period'])**(params['d'] * params['tissue_shape'][0]**2)) for nu in nus]
    plt.plot(nus, analytic, '--', label = 'Analytic')

    plt.errorbar(nus, mean_time_in_AF, yerr=[i/(params['realisations'])**0.5 for i in std_devs], xerr=None, fmt='x', label = 'Data from us')

    plt.xlabel('Fraction of vertical connections, '+r'$\nu$')
    plt.ylabel('Mean time in AF/ Risk of AF')
    plt.title('Realisations: {0}'.format(params['realisations']))
    plt.legend()
    plt.show()

################# INVESTIGATING ATTRACTOR DYNAMICS ###############

def generate_single_substrate_data(nu):

    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    tt = TimeTracker(tmax=params['tmax'])
    runner = TimeTracker(tmax=params['realisations'])
    
    myocardium = Myocardium(params['tissue_shape'], nu, params['d'], 
                            params['e'], params['refractory_period'])
	
    for i in runner:
        print "Beginning simulation number {0}".format(i)
        total_activity = TotalActivity()
        file_name = "/SS-nu-{1}-Run-{0}".format(i, nu)
        for time in tt:
            if time%params['heart_rate'] == 0:
                myocardium.evolve(pulse=True)
            else:
                myocardium.evolve()
            activity = myocardium.number_of_active_cells()
            total_activity.record(activity)
        with open(dirname + file_name, 'w') as fh:
            pickle.dump(total_activity.activity, fh)
        myocardium.reset()

# Is there a way to save the initial myocardium, so that once this function has been run, one could go back and generate more data for other nus?
def generate_single_substrate_data_aging_tissue(nus):

    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    sp = StatePickler()
    tt = TimeTracker(tmax=params['tmax'])
    runner = TimeTracker(tmax=params['realisations'])
   
    myocardium = Myocardium(params['tissue_shape'], nus[0], params['d'], 
                            params['e'], params['refractory_period'])

    
    for nu in nus:

        sp.pickle_state(out_dir = dirname, myocardium = myocardium, 
                        random_state = np.random.get_state(), t=0) 

        for i in runner:
            print "Beginning simulation number {0} for nu = {1}".format(i, nu)
            total_activity = TotalActivity()
            file_name = "/SS-nu-{1}-Run-{0}".format(i, nu)
            for time in tt:
                if time%params['heart_rate'] == 0:
                    myocardium.evolve(pulse=True)
                else:
                    myocardium.evolve()
                activity = myocardium.number_of_active_cells()
                total_activity.record(activity)
            with open(dirname + file_name, 'w') as fh:
                pickle.dump(total_activity.activity, fh)
            myocardium.reset()
        if nu != nus[-1]:
            myocardium.age_tissue(nu, nus[nus.index(nu) + 1])

# May need to rethink this function - want to plot multiple basins of attraction plots but for on substrate with multiple nus (mimicking aging tissue)
def plot_basins_of_attraction(files, nu, avg=False):
    """ Plot multiple realisations at a given nu on the same axis. """

    if avg is False:
        ax = plt.gca()
        ax.clear()
    elif avg is True:
        fig, (ax, avg_ax) = plt.subplots(2, 1, sharex=True)

    num_colours = len(glob.glob(files))
    cmap = plt.get_cmap('rainbow')

    ax = prepare_axes(ax, title=r"$\nu={0}$".format(nu),
                      ylabel="$Fraction\ of\ time\ in\ configurations\ with\ activity\ A$")

    i = 1
    avgs = None
    for fname in glob.glob(files):
        print "Working on {0}".format(fname)
        with open(fname, 'r') as fh:
            time_act = pickle.load(fh)
            if isinstance(time_act, list):
                time_act = np.array(time_act)
            try:
                activity = time_act[0, :]
            except IndexError:
                activity = time_act
            time_at_activity = np.bincount(activity) / float(activity.size)
            all_activity_vals = np.arange(np.max(activity)+1)
            ax.plot(all_activity_vals, time_at_activity, '-',
                    c=cmap(1.*i/num_colours))
            if avg is True:
                if avgs is None:
                    avgs = time_at_activity
                else:
                    if avgs.size < time_at_activity.size:
                        tmp_avgs = time_at_activity
                        tmp_avgs[:avgs.size] += avgs
                        avgs = tmp_avgs
                    else:
                        avgs[:time_at_activity.size] += time_at_activity
        i += 1

    if avg is True:
        avg_ax = prepare_axes(avg_ax, xlabel="$Activity\ A$", 
                              ylabel="$Average\ fraction\ of\ time\ in\ configurations\ with\ activity\ A$")
        avg_ax.plot(avgs / i)

    plt.show()

#Want function to compute the boundaries here.

def probability_of_entering_attractor(activity, x):
    # NOTE: These values depend on how we define the width of each attractor. 
    #       May want to call function to calculate appropriate boundaries here.
    high_bound_attractor_1 = 205
    low_bound_attractor_2 = 537

    ### Calculating all times at which system is in sinus rhythm ###
    a = 0
    leavesfib = [0]  #Assume system starts in sinus rhythm (as it does)
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]
    
    def times_in_sinus_rhythm(a,activity,treshold):
        while a != None:
            try:
                i = enters_fib(a, activity, threshold)
                a = i
                j = leaves_fib(a, activity, threshold)
                leavesfib.append(j) 
                a = j
            except IndexError:
                return
            
    times_in_sinus_rhythm(a,activity,threshold)
    #################################################################
    ### Calculating times when activity = x and system has just come from sinus rhythm ###
    t0 = []
    def activity_is_x(t, activity):
        """ Searches activity list from time t for first occurance of activity = x.
            Assume we have ensured system is coming from sinus rhythm (equive attractor 1)."""
        for i in range(t,len(activity)):
            if activity[i] == x:
                return i
            if i+1 == len(activity):
                raise IndexError 
    
    for i in leavesfib:
        try:
            t0.append(activity_is_x(i, activity))
        except IndexError:
        	pass
    #######################################################################################
    ### Check whether the activity proceeds to attractor 1 or 2 first, and append the  ###
    ### result to the appropriate list.                                                ###

    P1 = []
    P2 = []
    if t0 != []:
    	for t in t0:
            try:
                att1 = cross_boundary(t, activity, high_bound_attractor_1, 'above')
                att2 = cross_boundary(t, activity, low_bound_attractor_2, 'below')
                # Deciding which attractor is entered first,
                if att2 < att1:
                    P1.extend([0])
                    P2.extend([1])
                elif att1 < att2:
                    P1.extend([1])
                    P2.extend([0])
            except IndexError:
                # From time t, either system only enters one of the attractors or it enters neither,
                try:
                    att1 = cross_boundary(t, activity, high_bound_attractor_1, 'above')
                    P1.extend([1])
                    P2.extend([0])
                except IndexError:
                    # System doesn't transition to attractor 1, meaning it either transitions to attractor 2 or neither, 
                    try:
                        att2 = cross_boundary(t, activity, low_bound_attractor_2, 'below')
                        P1.extend([0])
                        P2.extend([1])
                    except IndexError:
                        pass    # Discard this value of x as not enough time elapsed for system to reach attractor.
    
    return P1, P2

def plot_prob_vs_x(nu):
    """ 
    Loads or Generates data for probability that the state transitions to attractor 1
    or attractor 2 for a range of initial activity values, then plots that data.
    This is designed to show that there are indeed attractors of the dynamics.
    """
    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    perc = 70

    try:
        data = pickle.load(open(dirname + '/prob-trans-curve-{0}-{1}-{2}'.format(nu, params['realisations'],perc), "r"))
        prob_attr_1 = data[0]
        prob_attr_2 = data[1]
        
    except:
        high_bound_attractor_1 = 205
        low_bound_attractor_2 = 537
        diff = low_bound_attractor_2 - high_bound_attractor_1
        data_points = 20

        xs = [i for i in range(high_bound_attractor_1, low_bound_attractor_2, diff/data_points)]

        prob_attr_1 = []
        prob_attr_2 = []

        for x in xs:
            print "Generating data for %i/%i" %(xs.index(x),len(xs))
            P1 = []
            P2 = []
            for i in range(0,params['realisations']):
                activity = pickle.load(open(dirname + "/SS-nu-{1}-Run-{0}".format(i, nu), "r"))
                p1, p2 = probability_of_entering_attractor(activity, x)
                ### For each x, Want to extend P1 for each i, then want to calculate the sum/length 
                ### which gives the 'probability' of transition to attractor 1.
                P1.extend(p1)
                P2.extend(p2)

            if P1 != []:
                prob_attr_1.extend( [(float(np.sum(P1))/len(P1), x)] )
            if P2 != []:
                prob_attr_2.extend( [(float(np.sum(P2))/len(P2), x)] )

        with open(dirname + '/prob-trans-curve-{0}-{1}-{2}'.format(nu, params['realisations'],perc),'w') as fh:
            pickle.dump(np.array([prob_attr_1, prob_attr_2]), fh)

    plt.plot( [j for (i,j) in prob_attr_1], [i for (i,j) in prob_attr_1], 'bo-', label = 'P1')  
    plt.plot( [j for (i,j) in prob_attr_2], [i for (i,j) in prob_attr_2] ,'ro-', label = 'P2')

    plt.title('Probability for transition to each attractor \n of the dynamics if activity starts at value x \n for simulation where nu = {0}, with {1} realisations.'.format(nu, params['realisations']))
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def plot_multiple_prob_vs_x(nu):
    """ 
    Loads or Generates data for probability that the state transitions to attractor 1
    or attractor 2 for a range of initial activity values, then plots that data.
    This is designed to show that there are indeed attractors of the dynamics.
    """
    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    percs = [70,80,90]    #Percentages associated with widths of boundaries
    high_bounds_attractor_1 = [205,205,205]
    low_bounds_attractor_2 = [501,520,537]

    n = len(2*percs)
    colour=iter(plt.cm.rainbow(np.linspace(0,1,n)))

    for perc in percs:
        try:
            data = pickle.load(open(dirname + '/prob-trans-curve-{0}-{1}-{2}'.format(nu, params['realisations'],perc), "r"))
            prob_attr_1 = data[0]
            prob_attr_2 = data[1]

        except:
            high_bound_attractor_1 = high_bounds_attractor_1[percs.index(perc)]
            low_bound_attractor_2 = low_bounds_attractor_2[percs.index(perc)]
            diff = low_bound_attractor_2 - high_bound_attractor_1
            data_points = 20

            xs = [i for i in range(high_bound_attractor_1, low_bound_attractor_2, diff/data_points)]

            prob_attr_1 = []
            prob_attr_2 = []

            for x in xs:
                print "Generating data for %i/%i" %(xs.index(x),len(xs))
                P1 = []
                P2 = []
                for i in range(0,params['realisations']):
                    activity = pickle.load(open(dirname + "/SS-nu-{1}-Run-{0}".format(i, nu), "r"))
                    p1, p2 = probability_of_entering_attractor(activity, x)
                    P1.extend(p1)
                    P2.extend(p2)

                if P1 != []:
                    prob_attr_1.extend( [(float(np.sum(P1))/len(P1), x)] )
                if P2 != []:
                    prob_attr_2.extend( [(float(np.sum(P2))/len(P2), x)] )

            with open(dirname + '/prob-trans-curve-{0}-{1}-{2}'.format(nu, params['realisations'],perc),'w') as fh:
                pickle.dump(np.array([prob_attr_1, prob_attr_2]), fh)

        plt.plot( [j for (i,j) in prob_attr_1], [i for (i,j) in prob_attr_1], c = next(colour), linestyle='o-', label = 'P1-{0}%'.format(perc))  
        plt.plot( [j for (i,j) in prob_attr_2], [i for (i,j) in prob_attr_2], c = next(colour), linestyle='o-', label = 'P2-{0}%'.format(perc))

    plt.title('Probability for transition to each attractor \n of the dynamics if activity starts at value x \n for simulation where nu = {0}, with {1} realisations.'.format(nu, params['realisations']))
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

############ SURVIVAL CURVE ############

def gen_survival_curve_data(nu, realisations = params['realisations']):

    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])

    times_spent_in_fib = []
    episodes_counted = 0
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]

    for i in range(0, realisations):
        print "Working on realisation: ", i
        # print("Episodes counted:", episodes_counted)
        try:
            with open(dirname + "/SS-nu-{0}-Run-{1}".format(nu, i), "r") as fh:
                activity = pickle.load(fh)

        except:
            print("Data does not exist for; SS-nu-{0}-Run-{1}".format(nu, i))
            break

        #Work out (for each realisation) when the system enters and leaves fib.
        try:
            t = 0
            while True:    
                t_enters = enters_fib(t, activity, threshold)
                t_leaves = leaves_fib(t_enters, activity, threshold)
                # if type(t_enters) == type(None) or type(t_leaves) == type(None):
                #     file_name = "/SS-nu-{1}-Run-{0}".format(i, nu)
                #     plot_activity(dirname+ file_name)
                if type(t_enters) == type(None):
                    print "t_enters FAILED"
                elif type(t_leaves) == type(None):
                    print "t_leaves FAILED"
                dt = t_leaves-t_enters

                # If the list of integers representing the number of times a sim 'survived in fib' isn't 
                # as long as the new data set of points to be added, extend this list with 0's to be that size. 
                if len(times_spent_in_fib) < dt:        
                    a = dt - len(times_spent_in_fib)
                    for i in range(0, a):
                        times_spent_in_fib.extend([0])

                # Add 1 to each time 'bin' for each timestep that the sim remained in att.2. 
                for i in range(0, dt):
                    times_spent_in_fib[i] += 1
                episodes_counted += 1
                
                t = t_leaves

        except IndexError:
            try: 
                # Simulation terminates in fib.
                t_enters = enters_fib(t, activity, threshold)
                dt = len(activity) - t_enters                 #Sim ends in fib. so tmax - time it entered can be added to survival curve calc.
        
                if len(times_spent_in_fib) < dt:
                    a = dt - len(times_spent_in_fib)
                    for i in range(0, a):
                        times_spent_in_fib.extend([0])

                for i in range(0, dt):
                    times_spent_in_fib[i] += 1
                episodes_counted += 1
            except IndexError:
                #Simulation terminates in Sinus rhythm
                pass
        except ValueError:
            #Sim enters fib within last two heartbeats of simulation
            pass

        # Converting times_spent_in_fib to probability list:
        P = [float(i)/episodes_counted for i in times_spent_in_fib]

        out_loc = dirname + '/Survival-Curve-{0}-{1}'.format(nu, realisations)
        with open(out_loc, 'w') as fh:
            pickle.dump(P, fh)

def plot_survival_curve(nu, realisations = params['realisations']):
    
    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    try:
        with open(dirname + '/Survival-Curve-{0}-{1}'.format(nu, realisations), 'r') as fh:
            P = pickle.load(fh)
        
    except:
        print("Generating data for Survival-Curve-{0}-{1}".format(nu, realisations))
        gen_survival_curve_data(nu = nu)
        with open(dirname + '/Survival-Curve-{0}-{1}'.format(nu, realisations), 'r') as fh:
            P = pickle.load(fh)

    ax = plt.gca()
    ax = prepare_axes(ax, title = 'Survival Curve,' +r' $\nu =$'+'${0}$'.format(nu), 
                      ylabel = 'Probability of remaining in fibrillation',
                      xlabel = 'Time spent in fibrillation')
    ax.plot([i for i in range(0,len(P))], P, '--')
    plt.show()

def plot_multiple_survival_curves(nus):

    dirname = 'Single-Substrate-tmax-{0}-d-{1}'.format(params['tmax'], params['d'])
    realisations = params['realisations']

    ax = plt.gca()
    ax = prepare_axes(ax, title = 'Survival Curve', 
                      ylabel = 'Probability of remaining in fibrillation',
                      xlabel = 'Time spent in fibrillation')
    n = len(nus)
    colour=iter(plt.cm.brg(np.linspace(0,0.9,n)))

    for nu in nus:

        try:
            with open(dirname + '/Survival-Curve-{0}-{1}'.format(nu, realisations), 'r') as fh:
                P = pickle.load(fh)
            print("Succesfully loaded survival curve for nu = ", nu)
        except:
            print("Generating survival curve for nu = ", nu)
            gen_survival_curve_data(nu, realisations)
            with open(dirname + '/Survival-Curve-{0}-{1}'.format(nu, realisations), 'r') as fh:
                P = pickle.load(fh)
        c = next(colour)
        ax.plot([i for i in range(0,len(P))], P, c=c, label = r' $\nu =$'+'${0}$'.format(nu))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=6, fancybox=True, shadow=True)
    plt.show()

############ MISC PLOTTING #############

def plot_activity(files):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = prepare_axes(ax)

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
        
    plt.show()

def plot_next_activity(fname):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = prepare_axes(ax) 

    with open(fname, 'r') as fh:
        time_act = pickle.load(fh)
        if isinstance(time_act, list):
            time_act = np.array(time_act)
        try:
            activity = time_act[0, :]
        except IndexError:
            activity = time_act

        ax.scatter(activity[:-1], activity[1:])

    plt.show()

if __name__ == "__main__":
    # generate_risk_curve_data()
       
    nus = [0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]       
    plot_multiple_survival_curves(nus)

    # nus = [0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]
    # generate_single_substrate_data_aging_tissue(nus)

    # plot_basins_of_attraction('./Single-Substrate-tmax-100000.0-d-0.01/SS-nu-0.09-Run-*', nu = 0.09, avg = True)
    
    # plot_prob_vs_x(nu = 0.05)

    # plot_activity('./Single-Substrate-tmax-100000.0-d-0.01/SS-nu-0.07-Run-10')
    # plot_next_activity('./Single-Substrate-tmax-100000.0-d-0.01/SS-nu-0.1-Run-3')