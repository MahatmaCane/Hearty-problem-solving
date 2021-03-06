import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
from AFTools import *
from TPM import TPM
import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 

params = dict(realisations = 40, tmax = 10e4, heart_rate = 220, 
              tissue_shape = (200,200), d = 0.05 , e = 0.05, 
              refractory_period = 50)

plt.ion()

def argand_eigenvalues(filepaths, nu, step=1):

    """ Plot Argand diagram of eigenvalues. """

    tpm = TPM(filepaths, nu, step=step)
    eig_with_largest_mod = np.max(np.absolute(tpm.eigenvalues))
    fig, (ax) = plt.subplots(1, 1)
    fig.suptitle(r"$\Delta t = {0}$".format(step))
    ax.scatter(tpm.eigenvalues.real, tpm.eigenvalues.imag, alpha=0.5, linewidths=0,
                c = (np.absolute(tpm.eigenvalues) == eig_with_largest_mod))
    ax.grid(True)
    ax.set_title(r"""$Argand\ Diagram\ of\ TPM\ e'vals,\ \nu={},\ max\lbrace \vert \lambda_i \vert \rbrace = {:.17f}$""".format(nu, eig_with_largest_mod))
    plt.show()

def plot_mod_eigenvalues(filepaths, nu, step=1, block=False):

    tpm = TPM(filepaths, nu, step=step)
    mod_eigs = tpm.get_mod_eigvals()
    eigenvalue_with_largest_mod = np.max(mod_eigs)
    fig, (ax) = plt.subplots(1, 1)
    ax = prepare_axes(ax, title=r"$\nu = {0}$".format(nu), xlabel=r"$i$",
                      ylabel=r"$\vert \lambda_i \vert$")
    ax.scatter(range(np.size(mod_eigs)), mod_eigs, alpha=0.4, linewidths=0,
               c = mod_eigs == eigenvalue_with_largest_mod)
    fig.suptitle(r"$\Delta t = {0}$".format(step))
    plt.show(block=block)

def plot_changing_mod_eigenvalues(files = [], nus = [], step = 1):

    fig, (ax) = plt.subplots(1, 1)
    ax = prepare_axes(ax, title=None, xlabel=r"Eigenvalue Rank, $i$",
                      ylabel=r"Magnitude of $i^{th}$ Eigenvalue, $\vert \lambda_i \vert$")
    i = 0
    for filepaths in files:
        tpm = TPM(filepaths, nus[i], step=step)
        mod_eigs = tpm.get_mod_eigvals()
        eigenvalue_with_largest_mod = np.max(mod_eigs)
        ax.plot(range(np.size(mod_eigs)), mod_eigs, 'o', alpha=0.4, label = r"$\nu = {0}$".format(nus[i]))
        i +=1
    plt.legend()
    plt.show()

def plot_eigenvector_matrix(filepaths, nu, step=1):

    tpm = TPM(filepaths, nu, step=step)
    fig, (real_ax, imag_ax) = plt.subplots(1, 2)
    real_im = real_ax.imshow(tpm.eigenvector_matrix.real, cmap="RdBu_r")
    fig.colorbar(real_im)
    imag_im = imag_ax.imshow(tpm.eigenvector_matrix.imag, cmap="RdBu_r")
    fig.colorbar(imag_im)
    plt.show(block=True)

def plot_degrees_activity(filepaths, nu, step=1):

    tpm = TPM(filepaths, nu, step=step)
    fig, (in_ax, out_ax) = plt.subplots(1, 2)
    fig.suptitle(r"$\Delta t={0},\ \nu={1}$".format(step, nu))
    prepare_axes(in_ax, xlabel=r"$Activity$", ylabel=r"$In-degree$")
    prepare_axes(out_ax, xlabel=r"$Activity$", ylabel=r"$Out-degree$")
    adj = tpm.tpm > 0.
    out_degs = adj.sum(axis=0)
    in_degs = adj.sum(axis=1)
    out_ax.scatter(range(out_degs.size), out_degs)
    in_ax.scatter(range(in_degs.size), in_degs)
    plt.show(block=True)

def plot_eigenvector(rank, filepaths, nu, step=1):

    tpm = TPM(filepaths,nu,step = step)
    eigval, eigvec = tpm.get_eig_val_and_associated_vec(rank)
    fig, (real_ax, imag_ax) = plt.subplots(1, 2)
    real_ax.scatter(range(np.size(eigvec)), eigvec.real, linewidths=0, alpha=0.4)
    imag_ax.scatter(range(np.size(eigvec)), eigvec.imag, linewidths=0, alpha=0.4)
    real_ax.grid(True)
    imag_ax.grid(True)
    fig.suptitle(r"$\Delta t={0},\ \nu = {1}$".format(step, nu))
    real_ax.set_title(r"$Re(eig'vector\ of\ largest\ eig'value)$")
    imag_ax.set_title(r"$Im(eig'vector\ of\ largest\ eig'value)$")
    plt.show()

def plot_diff_eigvals_vs_nu(i,j, patient):
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    nus = get_nus(dirname + '/')
    eigval_diffs = []
    for nu in nus:
        file_name = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
        filepaths = dirname + file_name
        tpm = TPM(filepaths, nu, step = 1)
        eigval_diffs.append( tpm.get_diff_absolute_eig_vals(i,j) )
    
    plt.plot(nus, eigval_diffs, 'o-')
    plt.xlabel(r"$\nu$", fontsize=25)
    plt.ylabel(r"$\sigma (P) \vert_{\nu}$", fontsize=25)
    plt.grid()
    plt.show(block=False)

def compute_eigval_diffs(patient, dirname):
    nus = get_nus(dirname + '/')
    print nus
    eigval_diffs = []
    for nu in nus:
        file_name = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
        filepaths = dirname + file_name
        tpm = TPM(filepaths,nu,step = 1)
        eigval_diffs.append( tpm.get_diff_absolute_eig_vals(1,2) )	
    return nus, eigval_diffs

def plot_diff_eigvals_multi_patient(rank_diff = [], patients = []):
    
    if len(patients) <= 1:
    	print "Must include more than 1 patient"
    	return

    k = 0
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','Z','W','X','Y','Z']
    
    delta1 = ["A", "B", "C", "D", "E", "F", "G", "H", "X"]
    delta5 = ["I", "J", "K", "L", "M", "N", "P", "Q", "T"]

    for patient in patients:
    	if patient in delta1:
    	    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], 0.01, params['e'])
    	elif patient in delta5:
    		dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], 0.05, params['e'])
        nus = get_nus(dirname)
        for (i,j) in rank_diff:
            eigval_diffs = []
            for nu in nus:
                file_name = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
                filepaths = dirname + file_name
                tpm = TPM(filepaths,nu,step = 1)
                eigval_diffs.append( tpm.get_diff_absolute_eig_vals(i,j) )

        plt.plot(nus, eigval_diffs, 'o-', label = 'Patient {0}'.format(alphabet[k]))
        k += 1

    plt.legend( bbox_to_anchor=(1.005 , 1), loc=2, borderaxespad=0., fancybox=True, shadow=True, fontsize = 22)
    # plt.title("Difference between eigenvalues of specified rank as a function of" +r" $\nu$", fontsize = 22)
    plt.grid()
    plt.xlabel(r"$\nu$", fontsize = 25 , labelpad=17)
    plt.ylabel(r"$\sigma (P)\vert_{\nu}$", fontsize = 25)
    plt.show(block=False)

def compute_baseline(patient):
    """ Function which takes in a patient's name, generates the eigval_diffs data for ranks 1,2
        uses data points from min_safe_nu upwards to estimate a baseline height. """
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])  
    nus, eigval_diffs = compute_eigval_diffs(patient, dirname)
    if nus[-1] < 0.7:
        raise "Must use value of nu = 0.9 or higher for computing baseline"
    elif nus[-1] >= 0.7:
        baseline = eigval_diffs[-1]     # Cut list of eigval_diffs to only values to be used in computing baseline	    
    
    return baseline, nus, eigval_diffs

def baseline_change_indicator(patient, thresh = 0.0005, delta_nu = 0.05):
    """ Computes the value of nu at which the data points first cross the threshold,
        which is defined as a percentage increase, thresh, above the baseline. """

    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    #nus, eigval_diffs = compute_eigval_diffs(patient,dirname)
    baseline, nus, eigval_diffs = compute_baseline(patient)
    nus.reverse()
    eigval_diffs.reverse()							
    #baseline = compute_baseline(patient)
    max_change = baseline + thresh*baseline
    print "MAX CHANGE:", max_change

    for nu in nus:    #High to low
        i = nus.index(nu)
        if eigval_diffs[i] > max_change:
            print "NU", nu
            precision = nus[i] - nus[i-1]
            if precision > delta_nu:
                raise Exception("Not enough data points between nu = {0} and nu = {1}".format(nus[i],nus[i-1]))
            for nu1 in nus[i:]:
                j = nus.index(nu1)
                # If there is a sudden drop in nu
                if abs(eigval_diffs[j] - eigval_diffs[j-1]) > baseline/2.:
                    print "NU1", nu1
                    nu_till_transition = np.absolute(nus[j-1] - nu)
                    return nu, nu_till_transition, precision
    raise ValueError("Change not detected.")

def prob_transition(patients, threshs, min_delt_nus=[0.2], nbins=100, ran=[0, 0.1]):

    """
    Produces a normalised 2D histogram of bin size bin_size. Element (i, j) of
    the histogram is the probability that the transition from pure SR dynamics
    to paroxysmal AF occurs in a ``fibrosis step'' in the range

        [i*1/nbins, (i+1)*1/nbins)

    (note the half-open interval) given a percentage change threshs[j].

    Inputs:

    - patients:     List of single capital letters indicating which patients 
                    the histogram is to be produced from. Assumes patient 
                    directories exist in the current directory.
    - threshs:      List of floats in range [0, 1]. For the jth element thresh
                    in this list, the spectral gap as a function of nu will be
                    searched for a percentage increase of thresh. The bin
                    corresponding to the change in nu to the transition and 
                    the specified percentage increase will then increase in 
                    count by 1.
    - min_delt_nus: List of floats. Minimum required nu resolution in spectral
                    gap data. If one element then each thresh in threshs will
                    require an identical minimum resolution in spectral gap
                    data at the point of crossing threshold.
    - nbins:        int. Number of bins along y-axis of matrix.
    """

    print len(min_delt_nus)
    if len(min_delt_nus) == 1:
        print "Increase"
        min_delt_nus *= len(threshs)
    assert len(threshs) == len(min_delt_nus), "Could not broadcast."

    hist = np.zeros((nbins, len(threshs)), dtype=np.float64)
    # Need to think of a way to estimate errors! Have to account for both
    # precision in point of crossing of threshold and point of transition.
    #errors = np.zeros((nbins, len(threshs)))

    for patient in patients:
        delta_nus = []
        for j, (thresh, mdn) in enumerate(zip(threshs, min_delt_nus)):
            nu, nu_trans, error = baseline_change_indicator(patient,
                                                            thresh=thresh,
                                                            delta_nu=mdn)
            print "THRESH:", thresh
            print "DELTA NU:", nu_trans
            delta_nus.append(nu_trans)
        print threshs
        print delta_nus
        print "\n"
        H, xedge, yedge = np.histogram2d(delta_nus, threshs, bins=[nbins, len(threshs)],
                                         range=[ran,[min(threshs), max(threshs)]])
        hist += H

    # Normalise
    normalisation = hist.sum(axis=0, dtype=np.float64)
    normalisation[normalisation == 0.] = 1.
    hist /= normalisation

    del min_delt_nus

    return hist

def plot_prob_transition(patients, threshs, min_delt_nus=[0.2], nbins=100, ran=[0, 0.1]):

    fig, (ax) = plt.subplots(1, 1)
    p = prob_transition(patients, threshs, min_delt_nus, nbins, ran)
    cum_p = np.cumsum(p, axis=0)
    im = ax.pcolorfast(cum_p)
    plt.colorbar(im)
    ax.set_ylabel(r"$\Delta \nu$", fontsize=30)
    ax.set_xlabel(r"$\Delta$", fontsize=30)
    ax.set_title(r"$P(transition\ in\ \Delta \nu \vert \Delta)$", fontsize=30)
    ax.set_xticklabels(threshs)
    xax = ax.get_xaxis()
    # Set tick locations as centre of column corresponding to each change in gap
    #r = [i/float(2*len(threshs)) for i in xrange(2*len(threshs)) if i%2 == 1]
    #print r
    xax.set_ticks([])
    xax.labelpad = 20
    yax = ax.get_yaxis()
    yax.set_ticks([])
    yax.labelpad = 20
    #print np.linspace(min(ran), max(ran), nbins)
    #ax.set_yticklabels(["{0}".format(i) for i in np.linspace(min(ran), max(ran), nbins)])
    plt.show(block=False)
    return p

def get_rank_change(patient):
    """ Return the rank of the TPM as a function of nu"""
    delta1 = ['A','C','E','G', 'H']
    delta5 = ['I','J','K']

    if patient in delta1:
        dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], 0.01, params['e'])
        delta = 0.01
    elif patient in delta5:
    	dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], 0.05, params['e'])
        delta = 0.05
    nus = get_nus(dirname + '/')
    ranks = []
    for nu in nus:
        file_name = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
        filepaths = dirname + file_name
        tpm = TPM(filepaths, nu, step = 1)
        ranks.append(tpm.get_rank())

    return nus, ranks, delta

def plot_rank_change(patients = []):

    for patient in patients:
        nus, ranks, delta = get_rank_change(patient)
        plt.plot(nus, ranks, 'o-', label = 'Patient '+ patient + ', $\delta = {0}$'.format(delta))
    # plt.title()
    plt.legend()
    plt.xlabel(r"$\nu$", fontsize = 22, labelpad = 17)
    plt.ylabel("Rank of Transition Probability Matrix", fontsize = 22, labelpad = 17)
    plt.show()


def plot_basin_peak_vs_nu(patient_dir):

    nus = get_nus(patient_dir)
    upper_basin_peaks = []
    upper_basin_peak_locs = {}
    patient = patient_dir.split("-")[1]
    fig, (ax, occ_ax) = plt.subplots(2, 1)
    for nu in nus:
        fname = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
        filepaths = patient_dir + fname
        tpm = TPM(filepaths, nu)
        occ_dens_index = np.where(tpm.eigenvalues == np.max(tpm.eigenvalues))
        occ_vec = tpm.eigenvector_matrix[:, occ_dens_index[0]].real
        if np.sum(occ_vec <= 0.) == occ_vec.size:
            occ_vec *= -1.
        elif np.sum(occ_vec >= 0.) != occ_vec.size:
            print np.sum(occ_vec >= 0.)
            raise Warning("Occupancy density is not strictly non-negative")
        if occ_vec.size <= 250:
            upper_basin_peaks.append(0.)
            continue
        upper_basin_peaks.append(np.max(occ_vec[250:]))
        peak_loc = np.where(occ_vec == np.max(occ_vec[250:]))[0][0]
        print peak_loc
        upper_basin_peak_locs[nu] = peak_loc
    occ_ax.plot(upper_basin_peak_locs.keys(), upper_basin_peak_locs.values(),
                'x')
    prepare_axes(ax, xlabel=r"$\nu$", title="Patient " + patient,
                 ylabel="Eigenvector centrality of upper basin")
    ax.plot(nus, upper_basin_peaks, 'x-')
    plt.show()

if __name__ == '__main__':
    pass
