#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import h5py
from common_plotting import smash_style


h5 = h5py.File('vor.hdf5', 'r')

def visit(h5file):
    '''print the name of all the dataset in h5file'''
    def printname(name):
        print name
    h5file.visit(printname)

def integrated_Piy_vs_rapidity(event_id):
    int_vor = h5['event%s/integral_pt_phi/vor'%event_id][...]
    int_rho = h5['event%s/integral_pt_phi/rho'%event_id][...]
    rapidity = h5['mom/Y'][...]
    return rapidity, int_vor/int_rho



def plot_int_Piy():
    Y, integrated_polarization = integrated_Piy_vs_rapidity(0)

    plt.plot(Y, integrated_polarization)
    plt.xlabel(r'$rapidity$')
    plt.ylabel(r'$int P^y dp_x dp_y$')
    smash_style.set()
    plt.savefig('%s/Pi_int_pxpy.png'%fpath)
    plt.close()
    #plt.show()

def plot_2d_Piy():
    pass


if __name__ == '__main__':



