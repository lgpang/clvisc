'''Lattice QCD EOS for 2+1flavor from Wuppertal-Budapest Group 2014'''
#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import numpy as np
#import sympy as sp
from scipy.integrate import quad
import math
#from numba import jit
from scipy import interpolate

hbarc = 0.19732

class EosWB(object):
    '''EOS from WuppertalBudapest Group 2014.
    $p/T^4 = p(T*)/T*^4 + \int_{T_*}^{T} I(T')/T'^5 dT'$
    e = I(T) + 3p
    I(T)/T^5 is not analytically integratable, so numerical
    integration is used here'''
    def __init__(self, Tstar=0.214, P_over_Tstar4=1.842):
        self.Tstar = Tstar
        self.P_over_Tstar4 = P_over_Tstar4

    def trace_anomaly(self, T):
        '''trace anomaly as a function of temperature

        Args:
             T [GeV]: temperature 

        Returns:
            The value of I(T)/T^4= (e(T)-3*P(T))/T^4 in formula
        '''
        # hi, fi, gi are parameters for LatticeQCD Eos
        h0, h1, h2 = 0.1396, -0.18, 0.035
        f0, f1, f2 = 1.05, 6.39, -4.72
        g1, g2 = -0.92, 0.57

        t = T/0.2
        return math.exp(-h1/t-h2/(t*t))*(h0+f0*(math.tanh(f1*t+f2)+1.0) \
                /(1.0+g1*t+g2*t*t))


    def P_over_T4(self, T):
        '''pressure / T^4

        Args:
           T Symbol('T'): temperature
        Returns:
           P/T^4 = \int_{T*}^T I(T')/T'^5 dT' + P*/T*^4
        '''
        f = lambda x: self.trace_anomaly(x)/x
        # g = \int I(T')/T'^5 dT'
        #g = sp.integrate(f, x)
        #return g.subs(x, T) - g.subs(x, self.Tstar) + self.P_over_Tstar4
        f_intg = quad(f, self.Tstar, T)[0] + self.P_over_Tstar4
        return f_intg

    def E_over_T4(self, T):
        '''energy_density / T^4

        Args:
           T Symbol('T'): temperature
        Returns:
           E/T^4 = 3*P/T^4 + trace_anomaly
        '''
        return 3*self.P_over_T4(T) + self.trace_anomaly(T)

    def cs2_T(self, T):
        ''' speed of sound square '''
        pot4 = eos.P_over_T4(T)
        eot4 = 3*pot4 + self.trace_anomaly(T)
        return pot4/eot4

    def energy_density(self, T):
        return self.E_over_T4(T)*(T**4)/(hbarc**3)

    #@jit
    def create_table(self):
        T_table = np.linspace(0.03, 1.13, 2000)
        ed_table = np.empty_like(T_table)
        cs2_table = np.empty_like(T_table)
        for i, T in enumerate(T_table):
            ed_table[i] = self.energy_density(T)
            cs2_table[i] = self.cs2_T(T)

        #plt.plot(ed_table, cs2_table*ed_table)
        tck = interpolate.splrep(ed_table, T_table, s=0)
        ed_new = np.linspace(0.01, 1999.99, 199999)
        T_new = interpolate.splev(ed_new, tck, der=0)
        tck = interpolate.splrep(ed_table, cs2_table, s=0)
        cs2_new = interpolate.splev(ed_new, tck, der=0)

        cs2_ed0 = -(cs2_new[1]-cs2_new[0])/(ed_new[1]-ed_new[0])*ed_new[0] + cs2_new[0]

        # Set the eos for ed=0.0, where T=Tmin, cs2=1/3
        ed_new = np.insert(ed_new, 0, 0.0)
        T_new = np.insert(T_new, 0, 0.001)
        cs2_new = np.insert(cs2_new, 0, cs2_ed0)
        pre_new = ed_new*cs2_new
        
        return ed_new, pre_new, T_new


# create eos table ed, pr, T
eos = EosWB()

ed, pr, T = eos.create_table()

ed_start = 0.0
ed_step = 0.01
num_ed = 200000



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    eos = EosWB()
    ed, pr, T = eos.create_table()
    #pr = np.diff(pr[:])
    #plt.plot(ed[:100000], pr[:100000], 'r--')
    plt.plot(ed[:100], T[:100], 'r--')
    plt.plot(ed[:100], pr[:100], 'b--')

    T_test = T[1001]
    ed_test = ed[1001]

    plt.show()

