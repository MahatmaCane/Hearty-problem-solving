import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from AFTools import *

class TPM():
    def __init__(self, filepaths, nu, step = 1):
        """ Upon instanciating the TPM object, all files of a given nu for a patient should be used in
            generating it. If it has been generated all ready, the file should be loaded. """

        self.nu = nu
        self.step = step
        self.tpm = self.construct(filepaths)
        self.eigenvalues, self.eigenvector_matrix = self.get_all_eig_vals_vecs(filepaths)

    def construct(self, filepaths):
    	""" Constructs the TPM by either loading from existing file, or if no file exists, creating it. """

    	tpm_loc = os.path.dirname(filepaths) + "/TPM-{}.npy".format(self.nu)

        # Check to see if exists in directory already
        if not os.path.exists(tpm_loc):

            print "File {0} does not exist. Creating.".format(tpm_loc)
            activities = dict()

            dim = None
            for fname in glob.glob(filepaths):
                activity = np.genfromtxt(fname)
                # +1 ensures that matrix dimensions allow activity values of 0 and dim
                this_max_act = np.max(activity) + 1
                if dim is None:
                    dim = this_max_act
                elif this_max_act > dim:
                    dim = this_max_act
                activities[fname] = activity
            
            H = None
            for activity in activities.values():
                x = activity[:-self.step]
                y = activity[self.step:]
                (hist, xe, ye) = np.histogram2d(x, y, bins=(dim, dim),
                                                range=[[0, dim],[0, dim]])
                if H is None:
                    H = hist
                else:
                    H += hist

            # Normalise
            normalisation = np.sum(H, axis=0, dtype=np.float64)
            normalisation[normalisation == 0.] = 1.
            trans_prob = H/normalisation
            np.save(tpm_loc, trans_prob)

        else:
            # print "File exists"
            trans_prob = np.load(tpm_loc)

        return trans_prob

    def get_all_eig_vals_vecs(self, filepaths):
    	""" Method for generating all eigenvalues and eigenvectors by either loading from existing files, 
    	    or if no file exists, creating it. """
    	
    	eigvals_loc = os.path.dirname(filepaths) + "/TPM-{}-eigvals.npy".format(self.nu)
    	eigvecs_loc = os.path.dirname(filepaths) + "/TPM-{}-eigvecs.npy".format(self.nu)
    	
        if not os.path.exists(eigvals_loc):
        	print "File {0} does not exist. Creating.".format(eigvals_loc)
        	if not os.path.exists(eigvecs_loc):
        		print "File {0} does not exist. Creating.".format(eigvecs_loc)

        	eigenvalues, eigenvector_matrix = np.linalg.eig(self.tpm)
        	np.save(eigvals_loc, eigenvalues)
        	np.save(eigvecs_loc, eigenvector_matrix)

        else:
            # print "Files exist"
            eigenvalues = np.load(eigvals_loc)
            eigenvector_matrix = np.load(eigvecs_loc)

        return eigenvalues, eigenvector_matrix

    def get_eig_val_and_associated_vec(self, rank):
    	""" Method for obtaining an eigenvalue of specified rank and its associated eigenvector. """

        evalues = [np.absolute(i) for i in self.eigenvalues]
        evalues.sort()
        evalues.reverse()
        eigval = evalues[rank-1]
        eigvec = self.eigenvector_matrix[:, [np.absolute(i) for i in self.eigenvalues].index(eigval)]
        return eigval, eigvec

    def get_mod_eigvals(self):
        """ Method to obtain the absolute values of all eigenvalues. """

        return np.absolute(self.eigenvalues)

    def get_diff_absolute_eig_vals(self, i, j):
    	""" Method to obtain the absolute difference between eigenvalues of rank i and j. """

    	eigval1, eigvec1 = self.get_eig_val_and_associated_vec(i)
        eigval2, eigvec2 = self.get_eig_val_and_associated_vec(j)
        return np.absolute(eigval1) - np.absolute(eigval2)

    def show(self):
    	""" Display the TPM """
        fig = plt.figure(figsize=(6, 3.2))
        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(self.tpm)
        plt.title("TPM at "+r"$\nu = {0}$".format(self.nu))
        plt.xlabel(r"$a(t+1)$")
        plt.ylabel(r"$a(t)$")
        plt.colorbar(orientation='vertical')
        plt.show()

    def test_linear_independence_of_evecs(self):
    	""" The eigenvectors of any homogenous matrix are 
    	    linearly independent if its determinant is non-zero. """
        det = np.linalg.det(self.tpm)
        if det == 0:
        	return False
        if det != 0:
        	return True

    def get_rank(self):
        return np.linalg.matrix_rank(self.tpm)

    def qr_decomposition(self, output = 'r', show = False):
        """
        q: A matrix with orthonormal columns
        r: The upper-triangular matrix (has same rank as TPM)
        """
        q, r = np.linalg.qr(self.tpm, mode = 'complete')
        
        if show == True:
            fig = plt.figure(figsize=(6, 3.2))
            ax = fig.add_subplot(111)
            ax.set_title('colorMap')
            if output == 'r': 
                plt.imshow(r)
                plt.title("r matrix (same rank as TPM) \n from QR decomposition of TPM \n "+r"$\nu = {0}$".format(self.nu))
            elif output == 'q':
            	plt.imshow(q)
            	plt.title("q matrix (orthonormal columns) \n from QR decomposition of TPM \n "+r"$\nu = {0}$".format(self.nu))
            plt.colorbar(orientation='vertical')
            plt.show()

        if output == 'r':
        	return r
        if output == 'q':
        	return q
