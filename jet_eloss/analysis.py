#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 01 Sep 2017 05:24:10 AM CEST

import matplotlib.pyplot as plt
import numpy as np
import os

fpaths = ['cent_20_30_etaos0.0/', 'cent_20_30_etaos0.0_with/', 'cent_20_30_etaos0.16/', 'cent_20_30_etaos0.16_with/']

ptspec = [np.loadtxt(os.path.join(fpaths[i], 'dNdYPtdPt_over_2pi_211.dat')) for i in range(4)]

d_ideal = ptspec[1][:, 1] - ptspec[0][:, 1]
d_visc = ptspec[3][:, 1] - ptspec[2][:, 1]
pt = ptspec[0][:, 0]

np.savetxt('ptspec_diff_ideal_vs_etas0p16.txt', np.array([pt, ptspec[0][:,1], ptspec[1][:,1], d_ideal,
                                               ptspec[2][:,1], ptspec[3][:,1], d_visc]).T,
            header='PT[GeV],  ideal{col1:nojet, col2:with_jet, col3:with-no}, \
            eta/s=0.16{col1:nojet, col2:with_jet, col3:with-no}', fmt='%.6e')

#plt.plot(pt, d_ideal, label=r'$\eta/s=0.0$')
#plt.plot(pt, d_visc, label=r'$\eta/s=0.16$')
#plt.show()
