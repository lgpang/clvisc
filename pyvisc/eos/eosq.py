#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 15 Jul 2016 23:27:42 CEST

import numpy as np
from scipy.interpolate import interp1d

# the InterpolatedUnivariateSpline works for both interpolation and extrapolation
from scipy.interpolate import InterpolatedUnivariateSpline


class EosQ(object):
    '''create first order phase transition eos;
    smooth at corner eh=0.45 GeV/fm^3 and eq=1.6 GeV/fm^3 where the
    cs2 jump from 0.15 to 0 and from 0 to 1.0/3.0. Here I follow
    the paper by HuiChao: http://arxiv.org/pdf/0712.3715v2.pdf
    cs2(e) is smoothed using a Fermi-distribution with width delta_e
    =0.1 GeV/fm^3 around eh and 0.2 GeV/fm^3 around eq'''
    def __init__(self):
        import os
        cwd, cwf = os.path.split(__file__)
        # data in eos_table/eosq are all MeV
        eosq = np.loadtxt(os.path.join(cwd, 'eos_table/eosq/eos_final_extended.dat'
                                      ), delimiter=',')
        size = 19930
        ed = eosq[:size, 1] * 0.001
        pr = eosq[:size, 2] * 0.001
        T =  eosq[:size, 3] * 0.001
        s =  (ed + pr) / T

        HRG_UPPER_BOUNDER = 21
        QGP_LOWER_BOUNDER = 77
        f_ed_hrg = interp1d(T[:HRG_UPPER_BOUNDER],
                            ed[:HRG_UPPER_BOUNDER])
        interp_order = 1

        f_ed_qgp = InterpolatedUnivariateSpline(
                ed[QGP_LOWER_BOUNDER:],
                ed[QGP_LOWER_BOUNDER:], k=interp_order)

        def mixed_phase(temperature):
            if temperature <= T[HRG_UPPER_BOUNDER]:
                return f_ed_hrg(temperature)
            elif temperature >= T[QGP_LOWER_BOUNDER]:
                return f_ed_qgp(temperature)
            else:
                return T[HRG_UPPER_BOUNDER + 1]
        self.f_ed = mixed_phase

        order = 1

        f_T_extra = lambda eds: (T[-1] - T[-2])/(ed[-1] - ed[-2]) * eds + T[-1]
        f_P_extra = lambda eds: (pr[-1] - pr[-2])/(ed[-1] - ed[-2]) * eds + pr[-1]
        f_S_extra = lambda eds: (s[-1] - s[-2])/(ed[-1] - ed[-2]) * eds + s[-1]

        # still have the extrapolation problem (const extrapolation)
        # this eos currently only works for < 500 GeV/fm^3
        self.f_T = InterpolatedUnivariateSpline(ed, T, k=order, ext=0)

        self.f_P = InterpolatedUnivariateSpline(ed, pr,k=order, ext=0)

        self.f_S = InterpolatedUnivariateSpline(ed, s, k=order, ext=0)


eos = EosQ()

ed = np.linspace(0, 1999.99, 200000)
pr = eos.f_P(ed)
T = eos.f_T(ed)
s = eos.f_S(ed)

f_P = eos.f_P
f_T = eos.f_T
f_S = eos.f_S
f_ed = eos.f_ed

ed_start = 0.0
ed_step = ed[1] - ed[0]
num_ed = 200000




if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.plot(ed, pr)
    plt.show()

