'''Lattice QCD EOS for 2+1flavor from Wuppertal-Budapest Group 2014'''
#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from scipy.integrate import quad
import math
from scipy import interpolate
import os

hbarc = 0.19732

class GlueBall(object):
    '''EOS from WuppertalBudapest Group 2014.
    GlueBall EOS, with first order phase transition'''
    def __init__(self, f_eostable):
        glueball_dat = np.loadtxt(f_eostable)

        # T in units [GeV], others are dimensionless
        T = 1.0E-3*glueball_dat[:,2]
        e_o_T4 = glueball_dat[:,0]
        p_o_T4 = glueball_dat[:,4]
        s_o_T3 = glueball_dat[:,3]

        hbarc3 = hbarc**3
        self.energy_density = e_o_T4 * T**4 / hbarc3       # in units GeV/fm^3
        self.pressure = p_o_T4 * T**4 / hbarc3            # in units GeV/fm^3
        self.entropy_density = s_o_T3 * T**3 / hbarc3     # in units fm^{-3}
        self.T = T

    def create_table(self):
        fT_ed = interpolate.interp1d(self.energy_density, self.T)
        ed_new = np.linspace(0.01, 1999.99, 199999)
        T_new = fT_ed(ed_new)
        fP_ed = interpolate.interp1d(self.energy_density, self.pressure)
        pre_new = fP_ed(ed_new)

        # Set the eos for ed=0.0, where T=Tmin, cs2=1/3
        ed_new = np.insert(ed_new, 0, 0.0)
        T_new = np.insert(T_new, 0, 0.001)
        pre_new = np.insert(pre_new, 0, 0.0)
        
        return ed_new, pre_new, T_new



# 
glueball_cwd, glueball_cwf = os.path.split(__file__)

glueball_datafile = os.path.join(glueball_cwd, 'glueball_eos.dat')

eos = GlueBall(glueball_datafile)

ed, pr, T = eos.create_table()

ed_start = 0.0
ed_step = 0.01
num_ed = 200000



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #pr = np.diff(pr[:])
    plt.plot(ed[:200], pr[:200], 'ro')

    T_test = T[1001]
    ed_test = ed[1001]
    print(ed[1000])
    
    plt.show()


