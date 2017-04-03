import glob
import os

def get_no_reals(dirname, nus):

    no_reals = []
    for nu in nus:
        files = dirname + "/Run-*-nu-{0}".format(nu)
        reals = None
        for fname in glob.glob(files):
            no_real = int(fname.split("-")[6])
            if reals is None:
                reals = no_real
            else:
                if no_real > reals:
                    reals = no_real
        no_reals.append(reals)
    return dict(zip(nus, no_reals))

def no_reals(dirname, nus):

    no_reals = []
    for nu in nus:
        files = dirname + "/Run-*-nu-{0}".format(nu)
        reals = len(glob.glob(files))
        no_reals.append(reals)
    return dict(zip(nus, no_reals))
