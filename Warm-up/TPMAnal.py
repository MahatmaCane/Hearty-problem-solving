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
    plt.show()

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
    plt.show()

def compute_baseline(patient, min_safe_nu = 0.6):
    """ Function which takes in a patient's name, generates the eigval_diffs data for ranks 1,2
        uses data points from min_safe_nu upwards to estimate a baseline height. """
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])  
    nus, eigval_diffs = compute_eigval_diffs(patient, dirname)
    lowest_nu_index = nus.index(min_safe_nu)          # Get index of lowest nu used in computing baseline
    eigval_diffs = eigval_diffs[lowest_nu_index:]     # Cut list of eigval_diffs to only values to be used in computing baseline	    
    
    return np.mean(eigval_diffs)

def baseline_change_indicator(patient, thresh = 0.0005, delta_nu = 0.05):
    """ Computes the value of nu at which the data points first cross the threshold,
        which is defined as a percentage increase, thresh, above the baseline. """

    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    nus, eigval_diffs = compute_eigval_diffs(patient,dirname)
    nus.reverse()
    eigval_diffs.reverse()							
    baseline = compute_baseline(patient)
    max_change = baseline + thresh*baseline

    for nu in nus:    #High to low
        i = nus.index(nu)
        if eigval_diffs[i] > max_change:
            if nus[i] - nus[i-1] > delta_nu:
                print "Not enough data points between nu = {0} and nu = {1}".format(nus[i],nus[i-1])
                return
            for nu1 in nus[i:]:
                j = nus.index(nu1)
                if np.absolute(eigval_diffs[j] - eigval_diffs[j-1]) > float(baseline)/2: #If there is a sudden drop in nu
                    nu_till_transition = np.absolute(nus[j-1] - nu)
                    return nu, nu_till_transition
    print "Change not detected."
    return 

def plot_rank_change(patient):
    """ Plot showing how the rank of the TPM changes with nu"""
    
    dirname = 'Patient-{0}-{1}-{2}-{3}'.format(patient, params['tmax'], params['d'], params['e'])
    nus = get_nus(dirname + '/')
    ranks = []
    for nu in nus:
        file_name = "/sim-patient-{0}-nu-{1}-Run-*".format(patient, nu)
        filepaths = dirname + file_name
        tpm = TPM(filepaths, nu, step = 1)
        ranks.append(tpm.get_rank())

    plt.plot(nus, ranks, 'o-')
    # plt.title()
    plt.xlabel(r"$\nu$", fontsize = 22, labelpad = 17)
    plt.ylabel("rank of transition probability matrix", fontsize = 22, labelpad = 17)
    plt.show()
