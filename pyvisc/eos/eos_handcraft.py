#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 15 Jul 2016 23:27:42 CEST

import numpy as np
from scipy.interpolate import interp1d, splrep, splev

# the InterpolatedUnivariateSpline works for both interpolation and extrapolation
from scipy.interpolate import InterpolatedUnivariateSpline

from eos import Eos

import matplotlib.pyplot as plt

class EosCraft(Eos):
    '''Hand crafted Eos by parameterization;
    used for regression problem in deep learning task;
    and EosQ tests for first order phase transition. '''
    def __init__(self, kind='eosq_modify', plot=False):
        super(EosCraft, self).__init__()
        if kind == 'eosq_modify':
            self.eosq_modify(plot)
        elif kind == 'eosl_modify':
            self.eosl_modify(plot)
        elif kind == 'parameterization':
            # construct eos by parameterization
            pass

    def eos_random(self, Xprim, R):
        '''Eq.2 in arXiv:1501.04042'''
        pass

    def eosq_modify(self, plot=False):
        '''modified eosq, whose pressure as a function of ed
        is given by s95p-pce for ed<0.5 and ed>1.9 (region out
        of mixed phase) and the mixed phase is by eosq'''
        eosq = Eos(5)
        eosl = Eos(1)
        ed = np.linspace(0.0, 2000.0, 200000, endpoint=True)
        # modify pressure vs ed in eosq
        self.pr = np.zeros_like(ed)
        hrg_phase = ed <= 0.5
        self.pr[hrg_phase] = eosl.f_P(ed[hrg_phase])
        mix_phase = (ed > 0.5) & (ed < 1.9)
        self.pr[mix_phase] = eosq.f_P(ed[mix_phase])
        qgp_phase = ed >= 1.9
        self.pr[qgp_phase] = eosl.f_P(ed[qgp_phase])
        self.ed = ed
        # set T by eosl
        self.T = eosl.f_T(ed)
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = 0.0
        self.ed_step = 0.01
        self.num_of_ed = 200000
        self.eos_func_from_interp1d()

        if plot:
            plt.plot(eosl.ed[:500], eosl.cs2[:500])
            plt.plot(self.ed[:500], self.cs2[:500])
            plt.show()

    def eosl_modify(self, plot=False):
        '''modified eosl, whose pressure as a function of ed
        is given by s95p-pce for crossover region and by eosq
        for other energy densities'''
        eosq = Eos(5)
        eosl = Eos(1)
        ed = np.linspace(0.0, 300.0, 200000, endpoint=True)
        # modify pressure vs ed in eosq
        self.pr = np.zeros_like(ed)
        hrg_phase = ed <= 0.5
        self.pr[hrg_phase] = eosl.f_P(ed[hrg_phase])
        mix_phase = (ed > 0.5) & (ed < 4.0)
        self.pr[mix_phase] = eosl.f_P(ed[mix_phase])
        qgp_phase = ed > 4.0
        # set pr=eosq.pr when eosq.pr > eosl.pr
        self.pr[qgp_phase] = eosq.f_P(ed[qgp_phase])
        self.ed = ed
        # set T by eosl
        self.T = eosq.f_T(ed)
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = 0.0
        self.ed_step = ed[1] - ed[0]
        self.num_of_ed = 200000
        self.eos_func_from_interp1d()

        if plot:
            plt.plot(eosl.ed[:3000], eosl.cs2[:3000])
            plt.plot(self.ed[:3000], self.cs2[:3000])
            plt.show()


if __name__ == '__main__':
    import pyopencl as cl
    test = EosCraft(kind='eosl_modify', plot=True)
    compile_options = []
    test.test_eos(0.1)
    test.test_eos(0.6)
    test.test_eos(2.0)
    test.test_eos(10.0)

    #eosl = Eos(1)
    #eosq = Eos(5)
    #plt.plot(eosl.ed, eosl.pr)
    #plt.plot(eosq.ed, eosq.pr)
    #plt.xlim(0, 10)
    #plt.ylim(0, 4)
    #plt.show()


