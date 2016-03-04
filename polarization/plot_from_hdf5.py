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
    events = range(1, 100)
    if kind == 'ideal':
        events = range(0, 50)
    if kind == '45':
        events = range(1, 9)

    for eid in events:
        if kind == 'visc' and (eid==14 or eid==15):
            '''these 2 events are bad events'''
            continue

        Y, integrated_polarization = integrated_Piy_vs_rapidity(h5, eid)
        if not np.isnan(integrated_polarization[0]):
            mean_int_pol = mean_int_pol + integrated_polarization
            num_good_events = num_good_events + 1.0
            print(integrated_polarization)
    mean_int_pol /= num_good_events

    return Y, mean_int_pol

def plot_visc_vs_ideal():
    h5_visc = h5py.File('vor_int_visc_cent20_25.hdf5', 'r')
    h5_ideal= h5py.File('vor_int_ideal_cent20_25.hdf5', 'r')

    h5_ideal_cent45_50 = h5py.File('vor_int_ideal_cent45_50.hdf5', 'r')

    Y, mean_pol_visc = int_Piy(h5_visc, kind='visc')
    Y, mean_pol_ideal = int_Piy(h5_ideal, kind='ideal')
    Y, mean_pol_ideal_45 = int_Piy(h5_ideal_cent45_50, kind='45')

    plt.errorbar(Y, mean_pol_visc, 0.1*mean_pol_visc, fmt='rs-', label=r'$\eta/s=0.08$')
    plt.errorbar(Y, mean_pol_ideal, np.sqrt(1.0/50.0)*mean_pol_ideal, fmt='bo-', label=r'ideal fluid')
    plt.errorbar(Y, mean_pol_ideal_45, np.sqrt(1.0/8.0)*mean_pol_ideal_45, fmt='g^-', label=r'ideal 45-50')
    plt.xlabel(r'$rapidity$')
    plt.ylabel(r'$P^y_{int}$')
    smash_style.set(line_styles=False)

    plt.legend(loc='upper center')
    plt.subplots_adjust(left=0.2)
    plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV,\ cent=20-25\%$')
    plt.savefig('Pi_ideal_vs_visc.pdf')
    #plt.show()
    plt.close()

def plot_2d_Piy():
    pass


if __name__ == '__main__':
    plot_visc_vs_ideal()



