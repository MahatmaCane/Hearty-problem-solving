import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
from AFModel import Myocardium
from AFTools import *
from AF import run
from TPM import TPM

params = dict(realisations = 40, tmax = 10e4, heart_rate = 220, 
              tissue_shape = (200,200), d = 0.05, e = 0.05, 
              refractory_period = 50)

#### Generic Functions Used Throughout ####

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
        unless the system does not leave fibrillation, then an IndexError is raised,
        unless the system enters fibrillation within the final 2 heartbeats, then a ValueError is raised."""

    if t > params['tmax'] - 2*params['heart_rate']: #Sim enters fib within the final 2 heartbeats and so has no time to leave.
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
                if np.mean(activity[i-1:i-1+params['heart_rate']]) <= params['tissue_shape'][0]:  
                    return i
        if i == test_activities[-2]:
            raise IndexError # Simulation terminated in fibrillatory state (Takes two heartbeats to return to sinus rhythm)

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
        except ValueError:
        	break

    #Creating a list of 0's and 1's for each time step to note whether in fibrillatory state or not. This simplifies calculating mean_time_in_AF.
    try:
        is_fibrillating = [0 for i in range(0,entersfib[0])]
    except IndexError:
        return 0  #System never enters fibrillation
    
    x = len(entersfib)
    y = len(leavesfib)

    # for k in range 0 to the length of the largest of entersfib or leaves fib. 
    for k in range(0, max(x,y)):
    	#Try and create a list of 1's for when the system enters fib to when it leaves, the append with a list of 0's.
        try:
            is_fibrillating.extend([1 for i in range(entersfib[k],leavesfib[k])])
            is_fibrillating.extend([0 for j in range(leavesfib[k],entersfib[k])])
        except IndexError:
            if is_fibrillating[-1] == 1:
                is_fibrillating.extend([0 for j in range(leavesfib[-1],int(params['tmax']))])
            else:
                is_fibrillating.extend([1 for i in range(entersfib[-1],int(params['tmax']))])
            break

    return np.mean(is_fibrillating)

#### Generate patient specific simulation data ####

def simulate_patient(patient, nus, state_file, realisations=range(params['realisations'])):
    """ state_file is initially None. This means in first call of run() a state file will be created.
        state_file is created with name: out_dir ( = os.path.dirname(dump_loc) ) + State-0
        Next call of run() want the state_file to be used to load the correct substrate.

        **Assumption: Location of defective cells remains unchanged when loading myocardium from state_file.
        **Assumption: When calling run() a new myocardium is created, and so myocardium.reset() is not needed.

    """

    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    
    for nu in nus:
        for i in realisations:

            print "Realisation {0}/{1}".format(i, max(realisations))

            file_name = "/sim-patient-{2}-nu-{1}-Run-{0}".format(i, nu, patient)
            dump_loc = dirname + file_name

            if os.path.exists(dump_loc):
                raise Exception, "Data already exists."

            run(params['tmax'], params['heart_rate'], params['tissue_shape'], nu, params['d'],
                params['e'], params['refractory_period'], False, dump_loc, None,
                state_file, True)
            
            if state_file == None:
                state_file = dirname + '/State-{0}'.format(0)

#### Time Series ####

def activity_time_series(patient, nu, run, fancy = False):
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    file_name = "/sim-patient-{0}-nu-{1}-Run-{2}".format(patient, nu, run)

    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]
    activity = [i for i in np.genfromtxt(dirname + file_name)]

    if fancy == True:
        transition_times = []
        t = 0
        try: 
            while True:
                t_enters = enters_fib(t, activity, threshold)
                transition_times.append(t_enters)

                t_leaves = leaves_fib(t_enters, activity, threshold)
                transition_times.append(t_leaves)

                if type(t_enters) == type(None):
                    print "t_enters FAILED"
                elif type(t_leaves) == type(None):
                    print "t_leaves FAILED"

                t = t_leaves

        except IndexError:
            try: 
                # Simulation terminates in fib.
                t_enters = enters_fib(t, activity, threshold)
                transition_times.append(t_enters)

            except IndexError:
                #Simulation terminates in Sinus rhythm
                pass
        except ValueError:
            #Sim enters fib within last two heartbeats of simulation
            pass

        b = 0
        counter = 1

        fig, (ax) = plt.subplots(1, 1)
        ax.plot(activity[:transition_times[0]], c = 'k', label = 'Sinus Rhythm')

        for i in range(1, 13):#len(transition_times[:13])):

            if b == 0:
                if i == 1:
                    ax.plot(range(transition_times[i-1],transition_times[i]), activity[transition_times[i-1]:transition_times[i]], color='orangered', label = 'Fibrillating')
                    b = 1
                elif i !=1:
                    ax.plot(range(transition_times[i-1],transition_times[i]), activity[transition_times[i-1]:transition_times[i]], color='orangered')
                    b = 1
            elif b == 1:
                ax.plot(range(transition_times[i-1],transition_times[i]), activity[transition_times[i-1]:transition_times[i]], c = 'k')
                b = 0

    elif fancy == False:
        fig, (ax) = plt.subplots(1, 1)
        ax.plot(activity, label = 'Patient: {0}'.format(patient))
    plt.xlabel("Time, $t$")
    plt.ylabel("Activity, $\mathcal{A}$")
    plt.legend()
    plt.grid()
    # plt.xlim([0,12400])
    plt.show()

#### Determining Critical Threshold - Risk Curve ####

def generic_model_risk_curve(realisations=params['realisations']):
    """ Generates the total_activity lists for each realisation of each nu, and saves them
        to a folder with appropriate file names. Saves the list of nus used."""

    kishanNus = [0.02, 0.04, 0.06, 0.08, 0.11,
                 0.13, 0.15, 0.17, 0.19, 0.21,
                 0.23, 0.25, 0.27, 0.29, 0.12,
                 0.14, 0.16, 0.18, 0.2, 0.22,
                 0.24, 0.26, 0.28, 0.3, 0.1]

    nus = [nu for nu in kishanNus if 0.1 <= nu <= 0.16]

    dirname = 'Generic-Risk-Curve-{0}-{1}-{2}'.format(params['d'], params['e'], params['tmax'])
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    try:
        mean_time_in_AF, std_devs = pickle.load(open(dirname + 'plot_data', 'w'))

    except:
        mean_time_in_AF = []
        std_devs = []

        for nu in nus:
            time_in_AF = []

            for i in range(*realisations):
                try:
                    activity = [i for i in np.genfromtxt(dirname + '/Run-{0}-nu-{1}'.format(i, nu))]
                    mean_time_in_fib = mean_time_fibrillating(activity)
                    time_in_AF.append(mean_time_in_fib)
                except:
                    print("Data for nu = {0}, realisation {1}, does not exist... generating data".format(nu,i))
                    
                    dump_loc = dirname + '/Run-{0}-nu-{1}'.format(i, nu)
                    run(params['tmax'], params['heart_rate'], params['tissue_shape'], nu, params['d'],
                        params['e'], params['refractory_period'], False, dump_loc, None,
                        None, True)

                    activity = [i for i in np.genfromtxt(dump_loc)]
                    mean_time_in_fib = mean_time_fibrillating(activity)
                    time_in_AF.append(mean_time_in_fib)

            mean_time_in_AF.append(np.mean(time_in_AF))
            std_devs.append(np.std(time_in_AF))

        with open(dirname + 'plot_data', 'w') as fh:
            pickle.dump(mean_time_in_AF, std_devs)


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

def patient_specific_risk_curve(patient = 'A', nus = [], plot = True):
    """
    Want to generate a risk curve for a given patient with enough data around the threshold.
    Must first run this function with patient_initial_nus to roughly determine the threshold.
    Then generate data for a dense set of nus around this threshold using the simulate_patient function.
    Then run this function with the full set of data calculated, in order to plot an accurate risk curve for given patient.

    Checks whether the plot has been created before, if so, plots it again by loading the data.
    If not, creates the plot data. When creating the plot data, checks whether all the necessary
    simulation data exists. If so, creates the plot. If not, prints message to state which file was missing.
    """
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    subdirname = '/Risk-Curve'
    if not os.path.exists(dirname+subdirname):
        os.makedirs(dirname+subdirname)

    try:
        mean_time_in_AF, std_devs = pickle.load(open(dirname + subdirname + 'plot_data', 'w'))

    except:
        for nu in nus:
            for i in range(0,params['realisations']):
                file_name = "/sim-patient-{0}-nu-{1}-Run-{2}".format(patient, nu, i)
                if not os.path.exists(dirname+file_name):
                    print "Incomplete data set, file_name: {0} does not exist".format(file_name)
                    return

        mean_time_in_AF = []
        std_devs = []    

        for nu in nus:
            print "Generating data for nu = {0}/{1}".format(nu,nus[-1])
            time_in_AF = []

            for i in range(0, params['realisations']):
                activity = [i for i in np.genfromtxt(dirname + "/sim-patient-{0}-nu-{1}-Run-{2}".format(patient, nu, i))]
                mean_time_in_fib = mean_time_fibrillating(activity)
                time_in_AF.append(mean_time_in_fib)

            mean_time_in_AF.append(np.mean(time_in_AF))
            std_devs.append(np.std(time_in_AF))

        with open(dirname + 'plot_data', 'w') as fh:
            pickle.dump(np.array([mean_time_in_AF, std_devs]), fh)    

    plt.errorbar(nus, mean_time_in_AF, yerr=[i/(params['realisations'])**0.5 for i in std_devs], xerr=None, fmt='x', label = 'Data from us')

    plt.xlabel('Fraction of vertical connections, '+r'$\nu$')
    plt.ylabel('Mean time in AF/ Risk of AF')
    plt.title('Realisations: {0}'.format(params['realisations']))
    plt.legend()
    plt.show()

#### Bifurcation of States ####

# Call basins_same_axes(files, nu, avg=False) from AFData.py

#### Flickering ####

def mean_frequency_of_episodes(patient, nu, realisations):
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    subdirname = '/Flickering-Data'
    
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]
    episodes_counted = 0

    for i in range(0, realisations):
        print "Realisation {0}/{1}".format(i,realisations)
        try:
            file_name = "/sim-patient-{0}-nu-{1}-Run-{0}".format(patient, nu, i)
            activity = [i for i in np.genfromtxt(dirname + file_name)]

        except:
            print("Data does not exist for; sim-patient-{0}-nu-{1}-Run-{2}".format(patient, nu, i))
            break

        try:
            t = 0
            while True:
                t_enters = enters_fib(t, activity, threshold)
                t_leaves = leaves_fib(t_enters, activity, threshold)

                if type(t_enters) == type(None):
                    print "t_enters FAILED"
                elif type(t_leaves) == type(None):
                    print "t_leaves FAILED"

                episodes_counted += 1
                
                t = t_leaves

        except IndexError:
            try: 
                # Simulation terminates in fib.
                t_enters = enters_fib(t, activity, threshold)
                episodes_counted += 1

            except IndexError:
                #Simulation terminates in Sinus rhythm
                pass
        except ValueError:
            #Sim enters fib within last two heartbeats of simulation
            pass

    result = float(episodes_counted)/realisations

    with open(dirname + subdirname + '/nu-{0}-realisations-{1}'.format(nu,realisations), 'w') as fh:
        pickle.dump(result, fh)

    return result

def plot_mean_frequency_episodes(patient, nus, realisations):
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    subdirname = '/Flickering-Data'

    mean_freqs = []

    for nu in nus:
        try:
            with open(dirname + subdirname +'/nu-{0}-realisations-{1}'.format(nu,realisations), 'r') as fh:
                data = pickle.load(fh)
            print "Succesfully loaded data for file_name: nu-{0}-realisations-{1}".format(nu,realisations)
            mean_freqs.append(data)
        except:
            print "Generating data for file_name: nu-{0}-realisations-{1}".format(nu,realisations)
            mean_freqs.append(mean_frequency_of_episodes(patient, nu, realisations))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = prepare_axes(ax)   
    ax.scatter(nus, mean_freqs)  
    plt.show()

#### Critical Slowing Down ####

def gen_survival_curve_data(patient, nu, realisations):
    """
    Generates and saves the survival curve data for given patient at a given nu.
    """

    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    subdirname = '/Survival-Curve'

    times_spent_in_fib = []
    episodes_counted = 0
    threshold = params['tissue_shape'][0]+0.05*params['tissue_shape'][0]

    for i in range(0, realisations):
        print "Working on realisation: ", i
        # print("Episodes counted:", episodes_counted)
        try:
            file_name = "/sim-patient-{0}-nu-{1}-Run-{2}".format(patient, nu, i)
            activity = [i for i in np.genfromtxt(dirname + file_name)]

        except:
            print("Data does not exist for file_name: {0}".format(file_name))
            break

        #Work out (for each realisation) when the system enters and leaves fib.
        try:
            t = 0
            while True:    
                t_enters = enters_fib(t, activity, threshold)
                t_leaves = leaves_fib(t_enters, activity, threshold)

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

                # Add 1 to each time 'bin' for each timestep that the sim remained in fib. 
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

        out_loc = dirname + subdirname +'-{0}-{1}.npy'.format(nu, realisations)
        np.save(out_loc, P)

def survival_curves_plot(patient, nus):

    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    subdirname = '/Survival-Curve'
    realisations = params['realisations']

    ax = plt.gca()
    ax = prepare_axes(ax, title = None, 
                      ylabel = 'Probability of remaining in fibrillation, $P_{fib}$',
                      xlabel = 'Time spent in fibrillation, $t$')
    n = len(nus)
    colour=iter(plt.cm.brg(np.linspace(0,0.9,n)))

    for nu in nus:
        surv_curve_loc = dirname + subdirname +'-{0}-{1}.npy'.format(nu, realisations)
        if not os.path.exists(surv_curve_loc):
            print "File {0} doesn't exist. Creating".format(surv_curve_loc)
            gen_survival_curve_data(patient, nu, realisations)
            P = np.load(surv_curve_loc)
        else:
            P = np.load(surv_curve_loc)
        c = next(colour)
        ax.plot([i for i in range(0,len(P))], P, c=c, label = r' $\nu =$'+'${0}$'.format(nu))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=6, fancybox=True, shadow=True, fontsize = 22)
    plt.show()

if __name__ == "__main__":
    pass

