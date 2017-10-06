#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 15 Jul 2016 23:27:42 CEST

import numpy as np
from scipy.interpolate import interp1d, splrep, splev

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
        size = 19953
        ed = eosq[:size, 1] * 0.001
        pr = eosq[:size, 2] * 0.001
        T =  eosq[:size, 3] * 0.001
        s =  (ed + pr) / T

        self.ed = ed
        self.pr = pr
        self.T = T

        HRG_UPPER_BOUNDER = 21
        QGP_LOWER_BOUNDER = 77
        f_ed_hrg = interp1d(T[:HRG_UPPER_BOUNDER+1],
                            ed[:HRG_UPPER_BOUNDER+1])
        interp_order = 1

        f_ed_qgp = InterpolatedUnivariateSpline(
                T[QGP_LOWER_BOUNDER-1:],
                ed[QGP_LOWER_BOUNDER-1:], k=interp_order)

        # notice that this one is only used to get efrz from Tfrz
        def mixed_phase(temperature):
            if temperature < T[HRG_UPPER_BOUNDER]:
                return f_ed_hrg(temperature)
            elif temperature > T[QGP_LOWER_BOUNDER]:
                return f_ed_qgp(temperature)
            else:
                return ed[HRG_UPPER_BOUNDER]

        self.f_ed = mixed_phase

        # still have the extrapolation problem (const extrapolation)
        # this eos currently only works for < 500 GeV/fm^3
        ### QGP_LOWER_BOUNDER + 200 is the safe range for linear intp
        N = size - 200

        u, indices = np.unique(pr, return_index=True)

        self.f_T = interp1d(ed[indices], T[indices])
        self.f_P = interp1d(ed[indices], pr[indices])
        self.f_S = interp1d(ed[indices], s[indices])

        def exponential_fit(x, a, b, c):
            return a*np.exp(-b*x) + c

        from scipy.optimize import curve_fit
        def fit_func(x, y):
            fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
            a, b, c = fitting_parameters
            return lambda next_x:exponential_fit(next_x, a, b, c)

        self.f_T_qgp = np.poly1d(np.polyfit(ed[N-10:], T[N-10:], 1))
        self.f_P_qgp = np.poly1d(np.polyfit(ed[N-10:], pr[N-10:], 1))
        self.f_S_qgp = np.poly1d(np.polyfit(ed[N-10:], s[N-10:], 1))



eos = EosQ()

ed = np.linspace(0, 1999.99, 200000)

ed_start = ed[0]
ed_step = ed[1] - ed[0]
num_ed = len(ed)

pr = np.empty_like(ed)
T = np.empty_like(ed)
s = np.empty_like(ed)

pr[ed < 400] = eos.f_P(ed[ed<400])
T[ed < 400] = eos.f_T(ed[ed<400])
s[ed < 400] = eos.f_S(ed[ed<400])

pr[ed >= 400] = eos.f_P_qgp(ed[ed>=400])
T[ed >= 400] = eos.f_T_qgp(ed[ed>=400])
s[ed >= 400] = eos.f_S_qgp(ed[ed>=400])


f_P = eos.f_P
f_T = eos.f_T
f_S = eos.f_S
f_ed = eos.f_ed

cs2 = np.gradient(pr, ed_step)



if __name__=='__main__':
    import matplotlib.pyplot as plt
    #for i in range(100):
    #    print('T=', T[i], ' ed=', f_ed(T[i]))
    print(f_T(0.142))
    #plt.plot(ed[:200000], s[:200000])
    #plt.plot(ed, T)
    #plt.plot(eos.ed, eos.pr)
    #plt.plot(ed[1000:10000], cs2[1000:10000])
    #plt.plot(ed, cs2)
    #plt.plot(ed[1000:10000], cs2[1000:10000])
    plt.show()

