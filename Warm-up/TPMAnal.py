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
    plt.title("Difference between eigenvalues of rank {0} and {1} for {2} as a function of".format(i,j,patient) +r" $\nu$")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\Delta$"+" $\lambda_{0} - \lambda_{1}$".format(i,j))
    plt.show(block=False)

def compute_eigval_diffs(patient, dirname):
    nus = get_nus(dirname + '/')
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
    
    delta1 = ['A','C','E','G', 'H']
    delta5 = ['I','J','K']

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

        plt.plot(nus, eigval_diffs, 'o-', label = 'Patient {0}, (m,n) = ({1},{2})'.format(alphabet[k], i,j))
        k += 1

    plt.legend( bbox_to_anchor=(1.005 , 1), loc=2, borderaxespad=0., fancybox=True, shadow=True, fontsize = 22)
    # plt.title("Difference between eigenvalues of specified rank as a function of" +r" $\nu$", fontsize = 22)
    plt.grid()
    plt.xlabel("Probability of vertical coupling, "+r"$\nu$", fontsize = 22 , labelpad=17)
    plt.ylabel("Eigenvalue difference, "+r"$\Delta$"+"$|\lambda_{m,n}|$", fontsize = 22)
    plt.show(block=False)

def compute_baseline(patient):
    """ Function which takes in a patient's name, generates the eigval_diffs data for ranks 1,2
        uses data points from min_safe_nu upwards to estimate a baseline height. """
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])  
    nus, eigval_diffs = compute_eigval_diffs(patient, dirname)
    if nus[-1] < 0.9:
        raise "Must use value of nu = 0.9 or higher for computing baseline"
    elif nus[-1] >= 0.9:
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
    im = ax.pcolorfast(p)
    plt.colorbar(im)
    ax.set_ylabel(r"$\Delta \nu$")
    ax.set_xlabel(r"$\Delta \lambda_{12}/\lambda_{12}$")
    ax.set_title(r"$P(transition\ in\ \Delta \nu \vert \Delta \lambda_{12}/\lambda_{12})$")
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


if __name__ == '__main__':
    pass
