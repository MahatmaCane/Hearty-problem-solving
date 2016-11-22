### Copywrite Jacob Swambo 10/10/2016 ###

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from AFModel import Myocardium
from AFTools import TimeTracker, TotalActivity

params = dict(realisations = 60, tmax = 10e4, heart_rate = 250, 
        tissue_shape = (200,200), nu = 0.25, d = 0.05, e = 0.05, 
        refractory_period = 50)

######### RISK CURVE ##############

def risk_curve_data_generator():

    nus = [0.02, 0.04, 0.06, 0.08, 0.11, 
           0.13, 0.15, 0.17, 0.19, 0.21, 
           0.23, 0.25, 0.27, 0.29, 0.12, 
           0.14, 0.16, 0.18, 0.2, 0.22, 
           0.24, 0.26, 0.28, 0.3, 0.1]

    dirname = 'Data-{0}-{1}-egram-activity'.format(params['d'], params['e'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for nu in nus:
        time_in_AF = []
        for i in range(0,params['realisations']):

            myocardium = Myocardium(params['tissue_shape'], nu, params['d'], 
                                    params['e'], params['refractory_period'])
            egram = TotalActivity()
            tt = TimeTracker(params['tmax'])
            egram.record(0, myocardium.number_of_active_cells())

            for time in tt:
                if time%params['heart_rate'] == 0:
                    myocardium.evolve(pulse=True)
                else:
                    myocardium.evolve()
                egram.record(time, myocardium.number_of_active_cells())


            with open(dirname + '/Run-{0}-{1}-{2}'.format(params['tmax'], nu, i),'w') as fh:
                pickle.dump(np.array([egram.activity]), fh)
            print "Run {0}/{1} saved for nu = {2}.".format(i+1, params['realisations'], nu)

    with open(dirname + '/Nus-{0}'.format(params['tmax']),'w') as fh:
        pickle.dump(np.array(nus), fh)

def is_fibrillating(activity):
    a = 0
    entersfib = []
    leavesfib = []
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]
    
    def enters(c):
        """" This function checks when the system first enters fibrillation from time c to time tmax """
        for i in range(c,len(activity[0])):

            if activity[0][i] >= threshold:
                return i
            if i+1 == len(activity[0]): 
                raise IndexError # Simulation Terminated in Sinus Rhythm

    def leaves(c):
        """ This function checks when the system first leaves fibrillation from time c to time tmax"""

        heart_beat_times = [j for j in range(0,len(activity[0])) if j%params['heart_rate'] == 0]
        b = 0

        for i in range(0, len(heart_beat_times)):  #Finds time of first heart beat after entering fibrillation and assigns b that value.
            if heart_beat_times[i] > c:            #Where c is the time that the system entered fibrillation
                b = heart_beat_times[i] 
                break

        for i in range(b, len(activity[0]), params['heart_rate']):

            if activity[0][i] <= threshold:
                if activity[0][i+params['heart_rate']] <= threshold:
                    # An additional constrain on defining sinus rhythm. The average activity is less than the width of the myocardium
                    # since the heartbeat is longer (250 > 200), i.e. there will be a period in sinus rhythm where there is 0 activity.
                    if np.mean(activity[0][i-1:i+params['heart_rate']]) <= params['tissue_shape'][0]:  
                        return i
            if i+1 == len(activity[0]):
                raise IndexError # Simulation Terminated in Fibrillatory state

    while a < params['tmax']:
        try:
            i = enters(a)
            entersfib.append(i)
            a = i
            j = leaves(a)
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

    dirname = 'Data-{0}-{1}-Risk_Curve'.format(params['d'], params['e'])
    nus = pickle.load(open(dirname + '/Nus-{0}'.format(params['tmax']), "r"))

    mean_time_in_AF = []
    std_devs = []

    for nu in nus:
        time_in_AF = []

        for i in range(0, params['realisations']):
            try:
                activity = pickle.load(open(dirname + '/Run-{0}-{1}-{2}'.format(params['tmax'], nu, i), "r"))
                is_fibrillating = np.array(activity[0]) >= (params['tissue_shape'][0]+0.05*params['tissue_shape'][0])
                time_in_AF.append(float(np.sum(is_fibrillating))/ params['tmax'])
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

###################################

########## INVESTIGATING ATTRACTOR DYNAMICS ############

def probability_of_state_transition(activity, x):
    """ For the first time when activity is equal to a given value, A = x, and when the dynamics proceeds to enter an attractor,
        this function generates a probability list (0s and 1s) for both the probability of the system evolving to 
        attractor 1 (associated with sinus rhythm) and to attractor 2 (associated with AF)."""

    high_bound_attractor_1 = 210
    low_bound_attractor_2 = 700

    def transition_to_attractor_2(t):
        """Assuming start time t, returns at what later time the system has entered attractor 2. """
        for i in range(t, len(activity[0])):
            if activity[0][i] >= low_bound_attractor_2:
                return i
            if i+1 == len(activity[0]):
                raise IndexError        #Simulation terminates in before transitioning to attractor 2.

    def transition_to_attractor_1(t):
        """Assuming start time t, returns at what later time the system has entered attractor 1. """
        for i in range(t, len(activity[0])):
            if activity[0][i] <= high_bound_attractor_1:
                return i
            if i+1 == len(activity[0]):
                raise IndexError        #Simulation terminates in before transitioning to attractor 2

    t0 = None
    ### Search through the activity list for the first time the activity is at x.
    ### Check whether the activity proceeds to attractor 1 or 2 first, and append the result to the appropriate list.
    for i in range(0,len(activity[0])):
        if activity[0][i] == x:
            t0 = i
            break

    P1 = []
    P2 = []
    if t0 != None:
        try:
            att1 = transition_to_attractor_1(t0)
            att2 = transition_to_attractor_2(t0)
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
                att1 = transition_to_attractor_1(t0)
                P1.extend([1])
                P2.extend([0])
            except IndexError:
                # System doesn't transition to attractor 1, meaning it either transitions to attractor 2 or neither, 
                try:
                    att2 = transition_to_attractor_2(t0)
                    P1.extend([0])
                    P2.extend([1])
                except IndexError:
                    pass    # Discard this value of x as not enough time elapsed for system to reach attractor.
    
    return P1, P2

def plot_prob_vs_x(nu = 0.13, realisations = 40):
    """ 
    Loads or Generates data for probability that the state transitions to attractor 1
    or attractor 2 for a range of initial activity values, then plots that data.
    This is designed to show that there are indeed attractors of the dynamics.
    """
    dirname = 'Data-{0}-{1}-egram-activity'.format(params['d'], params['e'])

    try:
        data = pickle.load(open(dirname + '/prob-trans-curve-{0}-{1}'.format(nu, realisations), "r"))
        prob_attr_1 = data[0]
        prob_attr_2 = data[1]
        
    except:
        xs = [210, 211, 212, 213, 214,
              215, 216, 217, 218, 219,
              220, 225, 250, 255, 260, 
              265, 270, 275, 280, 285, 
              290, 295, 300, 305, 310, 
              315, 320, 325, 330, 335, 
              340, 345, 350, 375, 400, 
              425, 450, 475, 500, 525, 
              550, 575, 600, 625, 650, 
              675, 700]

        prob_attr_1 = []
        prob_attr_2 = []

        for x in xs:
            print "Generating data for {0}/{1}".format(xs.index(x),len(xs))
            P1 = []
            P2 = []
            for i in range(0,realisations):
                activity = pickle.load(open(dirname + '/Run-{0}-{1}-{2}'.format(params['tmax'], nu, i), "r"))
                p1, p2 = probability_of_state_transition(activity, x)
                ### For each x, Want to extend P1 for each i, then want to calculate the sum/length which gives the 'probability' of transition to attractor 1
                P1.extend(p1)
                P2.extend(p2)

            if P1 != []:
                prob_attr_1.extend( [(float(np.sum(P1))/len(P1), x)] )
            if P2 != []:
                prob_attr_2.extend( [(float(np.sum(P2))/len(P2), x)] )

        with open(dirname + '/prob-trans-curve-{0}-{1}'.format(nu, realisations),'w') as fh:
            pickle.dump(np.array([prob_attr_1, prob_attr_2]), fh)

    # print "prob_attr_1: {0}".format(prob_attr_1)
    # print "prob_attr_2: {0}".format(prob_attr_2) 

    plt.plot( [j for (i,j) in prob_attr_1], [i for (i,j) in prob_attr_1], 'bo-', label = 'P1')  
    plt.plot( [j for (i,j) in prob_attr_2], [i for (i,j) in prob_attr_2] ,'ro-', label = 'P2')

    plt.title('Probability for transition to each attractor \n of the dynamics if activity starts at value x \n for simulation where nu = {0}, with {1} realisations.'.format(nu, realisations))
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

######### MISC PLOTTING ##########

def plot_activity(nu, i):
   
    dirname = 'Data-{0}-{1}-Risk_Curve'.format(params['d'], params['e'])
    activity = pickle.load(open(dirname + '/Run-{0}-{1}-{2}'.format(params['tmax'], nu, i), "r"))
    time = range(0, int(params['tmax'])+1)

    plt.plot(time,activity[0],'-')
    plt.title('Activity for ' + r'$\nu=$'+'${0}$'.format(nu))
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.show()

##################################

if __name__ == "__main__":

    # risk_curve_data_generator()
    # risk_curve_plot()
    # plot_activity(nu = 0.075, i = 1)
    # plot_prob_vs_x()
