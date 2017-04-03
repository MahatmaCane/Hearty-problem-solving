import glob
import os

def get_nus(directory):

    """Input: patient or directory? I think directory."""

    # Set of nus
    nus = set()

    for fname in glob.glob(os.path.dirname(directory) + "/sim-*"):
        nu = fname.split("-")[8]
        nus.add(float(nu))

    return sorted(nus)
