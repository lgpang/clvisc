#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class EosWBAnalytical(object):
    '''EOS from WuppertalBudapest Group 2014.
    $p/T^4 = p(T*)/T*^4 + \int_{T_*}^{T} I(T')/T'^5 dT'$
    e = I(T) + 3p'''
    def __init__(self, Tstar=0.214, P_over_Tstar4=1.842):
        self.Tstar = Tstar
        self.P_over_Tstar4 = P_over_Tstar4

    def trace_anomaly(self, T):
        '''trace anomaly as a function of temperature

        Args:
             T Symbol('T'): temperature symbol 

        Returns:
            The value of I(T)/T^4= (e(T)-3*P(T))/T^4 in formula
        '''
        # hi, fi, gi are parameters for LatticeQCD Eos
        h0, h1, h2 = 0.1396, -0.18, 0.035
        f0, f1, f2 = 1.05, 6.39, -4.72
        g1, g2 = -0.92, 0.57

        t = T / 0.2
        return sp.exp(-h1/t-h2/(t*t))*(h0+f0*(sp.tanh(f1*t+f2)+1.0) \
                /(1.0+g1*t+g2*t*t))


    def P_over_T4(self, T):
        '''pressure / T^4

        Args:
           T Symbol('T'): temperature
        Returns:
           P/T^4 = \int_{T*}^T I(T')/T'^5 dT' + P*/T*^4
        '''
        x = sp.Symbol('x')
        f = self.trace_anomaly(x)/x
        # g = \int I(T')/T'^5 dT'
        g = sp.integrate(f, x)
        print g

        return g.subs(x, T) - g.subs(x, self.Tstar) + self.P_over_Tstar4

    def E_over_T4(self, T):
        '''energy_density / T^4

        Args:
           T Symbol('T'): temperature
        Returns:
           E/T^4 = 3*P/T^4 + trace_anomaly
        '''
        return 3*self.P_over_T4(T) + self.trace_anomaly(T)

eos = EosWBAnalytical()

T = sp.Symbol('T')

print 'P/T^4=', eos.P_over_T4(T)

eot4 = eos.E_over_T4(T).subs(T, 0.5)
pot4 = eos.P_over_T4(T).subs(T, 0.5)

print pot4/eot4
