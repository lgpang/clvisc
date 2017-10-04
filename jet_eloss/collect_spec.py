#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 07 Sep 2017 09:10:00 AM CEST

import matplotlib.pyplot as plt
import numpy as np
import os

along_jet_direction = True

def average_spec(ebe_path):
    specs = []
    for i in range(100):
        if along_jet_direction:
            fname = os.path.join(ebe_path, 'event%s'%i, 'piplus_spec_along_jet.txt')
        else:
            fname = os.path.join(ebe_path, 'event%s'%i, 'piplus_spec_opposite_to_jet.txt')
        specs.append(np.loadtxt(fname))
    return np.array(specs).mean(axis=0)


def load_spec(fpath):
    if along_jet_direction:
        return np.loadtxt(os.path.join(fpath, 'piplus_spec_along_jet.txt'))
    else:
        return np.loadtxt(os.path.join(fpath, 'piplus_spec_opposite_to_jet.txt'))


fpaths = ['/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.0/',
          '/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.0_with/',
          '/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.16/',
          '/lustre/nyx/hyihp/lpang/jet_eloss/cent_20_30_etaos0.16_with/']


ptspec = [load_spec(fpaths[0]), average_spec(fpaths[1]), load_spec(fpaths[2]), average_spec(fpaths[3])]

print(ptspec)


d_ideal = ptspec[1][:, 1] - ptspec[0][:, 1]
d_visc = ptspec[3][:, 1] - ptspec[2][:, 1]
pt = ptspec[0][:, 0]

fout_name = 'ebe_ptspec_diff_ideal_vs_etas0p16_along_jet.txt'

if not along_jet_direction:
    fout_name = 'ebe_ptspec_diff_ideal_vs_etas0p16_opposite_to_jet.txt'

np.savetxt(fout_name, np.array([pt, ptspec[0][:,1], ptspec[1][:,1], d_ideal,
                                               ptspec[2][:,1], ptspec[3][:,1], d_visc]).T,
            header='PT[GeV],  ideal{col1:nojet, col2:with_jet, col3:with-no}, \
            eta/s=0.16{col1:nojet, col2:with_jet, col3:with-no}', fmt='%.6e')

plt.plot(pt, d_ideal, label=r'$\eta/s=0.0$')
plt.plot(pt, d_visc, label=r'$\eta/s=0.16$')
plt.show()
