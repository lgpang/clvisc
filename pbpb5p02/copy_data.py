#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 21 Apr 2016 16:09:22 CEST


import os
from subprocess import call

def copy(src, dest):
    files = ['dNdEta_mc_charged.dat',
            'bulk.dat', 'hydro.info', 'nbin.dat',
            'dN_over_2pidYptdpt_mc_211.dat',
            'dN_over_2pidYptdpt_mc_2212.dat',
            'dN_over_2pidYptdpt_mc_321.dat',
            'dN_over_2pidYptdpt_mc_charged.dat']

    if not os.path.exists(dest):
        os.makedirs(dest)

    for fname in files:
        fname = os.path.join('/lustre/nyx/hyihp/lpang/pbpb5p02/%s'%src, fname)
        try:
            call(['cp', fname, dest])
        except:
            print(fname, ' does not exist!')


def main():
    eid = range(100, 108)
    cent = [(0, 5), (5, 10), (10, 20), (20, 30),
            (0, 10), (0, 80), (10, 30), (30, 50)]

    for i, idx in enumerate(eid):
        cmin, cmax = cent[i]
        dest = os.path.join('pbpb2760', 'cent%s_%s'%(cmin, cmax))
        copy('event%s'%idx, dest)

main()
