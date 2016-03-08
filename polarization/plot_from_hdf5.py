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
    int_vor = h5['%s/integral_pt_phi/vor'%event_id][...]
    int_rho = h5['%s/integral_pt_phi/rho'%event_id][...]
    rapidity = h5['mom/Y'][...]
    return rapidity, int_vor/int_rho



def int_Piy(h5):
    '''get all the events in h5 and do average '''
    mean_int_pol = np.zeros(11)
    num_good_events = 0.0
    for eid in h5.keys():
        if 'event' in eid:
            Y, integrated_polarization = integrated_Piy_vs_rapidity(h5, eid)
            if not np.isnan(integrated_polarization[0]):
                mean_int_pol = mean_int_pol + integrated_polarization
                num_good_events = num_good_events + 1.0
                print(integrated_polarization)
    mean_int_pol /= num_good_events

    return num_good_events, Y, mean_int_pol



def plot_auau200_ideal_cent():
    '''auau200 polarization as a function of centrality '''
    h5_ideal_cent20_25 = h5py.File('vor_int_ideal_cent20_25.hdf5', 'r')
    h5_ideal_cent45_50 = h5py.File('vor_int_ideal_cent45_50.hdf5', 'r')
    h5_ideal_cent70_75 = h5py.File('vor_int_ideal_cent70_75.hdf5', 'r')
    n0, Y, mean_pol_0 = int_Piy(h5_ideal_cent20_25)
    n1, Y, mean_pol_1 = int_Piy(h5_ideal_cent45_50)
    n2, Y, mean_pol_2 = int_Piy(h5_ideal_cent70_75)

    plt.errorbar(Y, mean_pol_0, np.sqrt(1.0/n0)*mean_pol_0, fmt='rs-', label=r'auau200 20-25 $\eta/s=0.0$')
    plt.errorbar(Y, mean_pol_1, np.sqrt(1.0/n1)*mean_pol_1, fmt='bo-', label='auau200 45-50 $\eta/s=0.0$')
    plt.errorbar(Y, mean_pol_2, np.sqrt(1.0/n2)*mean_pol_2, fmt='g^-', label=r'auau200 70-75 $\eta/s=0.0$')
    plt.xlabel(r'$rapidity$')
    plt.ylabel(r'$P^y_{int}$')
    smash_style.set(line_styles=False)

    plt.legend(loc='upper center')
    plt.subplots_adjust(left=0.2)
    plt.title(r'$Au+Au\ \sqrt{s_{NN}}=200\ GeV,\ \eta/s=0$')
    plt.savefig('Pi_auau200.pdf')
    #plt.show()
    plt.close()




def plot_visc_vs_ideal():
    h5_visc = h5py.File('vor_int_visc0p08_cent20_25.hdf5', 'r')
    #h5_ideal= h5py.File('vor_int_ideal_cent20_25.hdf5', 'r')
    h5_ideal= h5py.File('vor_int_visc0p12_auau62p4_cent20_25.hdf5', 'r')
    h5_ideal_cent45_50 = h5py.File('vor_int_visc0p12_auau62p4_cent45_50.hdf5', 'r')

    h5_39 = h5py.File('vor_int_visc0p08_auau39_cent20_25.hdf5', 'r')

    n0, Y, mean_pol_visc = int_Piy(h5_visc)
    n1, Y, mean_pol_ideal = int_Piy(h5_ideal)
    n2, Y, mean_pol_ideal_45 = int_Piy(h5_ideal_cent45_50)
    n3, Y, mean_pol_3 = int_Piy(h5_39)

    plt.errorbar(Y, mean_pol_ideal_45, np.sqrt(1.0/n2)*mean_pol_ideal_45, fmt='g^', label=r'auau62.4 45-50 $\eta/s=0.12$')
    plt.errorbar(Y, mean_pol_ideal, np.sqrt(1.0/n1)*mean_pol_ideal, fmt='bo', label='auau62.4 20-25 $\eta/s=0.12$')
    plt.errorbar(Y, mean_pol_visc, np.sqrt(1.0/n0)*mean_pol_visc, fmt='rs', label=r'auau200 20-25 $\eta/s=0.08$')
    plt.errorbar(Y, mean_pol_3, np.sqrt(1.0/n3)*mean_pol_3, fmt='cd', label=r'auau39 20-25 $\eta/s=0.08$')
    plt.xlabel(r'$rapidity$')
    plt.ylabel(r'$P^y_{int}$')
    #smash_style.set(line_styles=False)
    smash_style.set()

    plt.ylim(0.0002, 0.005)
    plt.legend(loc='upper center')
    plt.subplots_adjust(left=0.2)
    plt.savefig('Pi_visc_cent.pdf')
    plt.show()
    plt.close()

def plot_2d_Piy():
    pass


if __name__ == '__main__':
    plot_visc_vs_ideal()
    #plot_auau200_ideal_cent()



