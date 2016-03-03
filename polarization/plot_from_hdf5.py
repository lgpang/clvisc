#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import h5py
from common_plotting import smash_style



def visit(h5file):
    '''print the name of all the dataset in h5file'''
    def printname(name):
        print name
    h5file.visit(printname)

def integrated_Piy_vs_rapidity(h5, event_id):
    int_vor = h5['event%s/integral_pt_phi/vor'%event_id][...]
    int_rho = h5['event%s/integral_pt_phi/rho'%event_id][...]
    rapidity = h5['mom/Y'][...]
    return rapidity, int_vor/int_rho



def int_Piy(h5, fpath='./', kind='ideal'):
    mean_int_pol = np.zeros(11)
    num_good_events = 0.0
    events = range(1, 50)
    if kind == 'ideal':
        events = range(0, 15)

    for eid in events:
        Y, integrated_polarization = integrated_Piy_vs_rapidity(h5, eid)
        if not np.isnan(integrated_polarization[0]):
            mean_int_pol = mean_int_pol + integrated_polarization
            num_good_events = num_good_events + 1.0
            print(integrated_polarization)
    mean_int_pol /= num_good_events

    return Y, mean_int_pol

def plot_visc_vs_ideal():
    h5_visc = h5py.File('vor_int.hdf5', 'r')
    h5_ideal= h5py.File('vor_int_ideal.hdf5', 'r')

    Y, mean_pol_visc = int_Piy(h5_visc, kind='visc')
    Y, mean_pol_ideal = int_Piy(h5_ideal, kind='ideal')

    plt.plot(Y, mean_pol_visc, label=r'$\eta/s=0.08$')
    plt.plot(Y, mean_pol_ideal, label=r'ideal fluid')
    plt.xlabel(r'$rapidity$')
    plt.ylabel(r'$P^y_{int}$')
    smash_style.set()

    plt.legend(loc='best')
    plt.subplots_adjust(left=0.2)
    plt.savefig('Pi_ideal_vs_visc.png')
    #plt.show()
    plt.close()

def plot_2d_Piy():
    pass


if __name__ == '__main__':
    plot_visc_vs_ideal()



