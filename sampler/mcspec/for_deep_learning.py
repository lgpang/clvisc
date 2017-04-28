#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style
import os
import pandas as pd

def create_figure_matrix(fpath, dat, pid=211, nsampling=2000, kind='Y', rapidity_window=1.0):
    E = dat[:,0]
    pz = dat[:,3]
    rapidity_col = 4
    if kind == 'Eta':
        rapidity_col = 6

    particle_type = (dat[:, 5]==pid)
    if pid == 'charged':
        particle_type = (dat[:, 5]==dat[:, 5])

    mid_rapidity = np.abs(dat[:, rapidity_col] ) < 0.5 * rapidity_window
    selected = particle_type & mid_rapidity

    pti = np.sqrt(dat[particle_type, 1]**2+dat[particle_type, 2]**2)

    #pti = pti[np.abs(Yi)<0.5*rapidity_window]
    pxi = dat[selected, 1]
    pyi = dat[selected, 2]

    hist2d, xedge, yedge = np.histogram2d(pxi, pyi, bins=30, range=[[-3, 3], [-3, 3]])

    comments = "#xedge=%s\n yedge=#%s"%(xedge, yedge)
    #comments = "px, py distribution"

    fname = os.path.join(fpath, 'hist2d_pxpy.txt')
    np.savetxt(fname, hist2d, header=comments)

def main(fpath, viscous_on, force_decay, nsampling):
    from subprocess import call, check_output
    cwd = os.getcwd()
    os.chdir('../build')
    call(['cmake', '..'])
    call(['make'])

    ns_str = '%s'%nsampling
    cmd = ['./main', fpath, viscous_on, force_decay, ns_str]

    proc = check_output(cmd)

    try:
        # used in python 2.*
        from StringIO import StringIO as fstring
    except ImportError:
        # used in python 3.*
        from io import StringIO as fstring

    #particle_lists = np.genfromtxt(fstring(proc))

    particle_lists = pd.read_csv(fstring(proc), sep=' ', header=None, dtype=np.float32, comment='#').values

    create_figure_matrix(fpath, particle_lists)

    #from mcspec import mcspec
    #mcspec(fstring(proc))

    os.chdir(cwd)

    #plot(fpath, particle_lists, nsampling = nsampling)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('usage:python for_deep_learning.py fpath')
        exit(0)

    fpath = sys.argv[1]
    #viscous_on = sys.argv[2]
    #force_decay = sys.argv[3]
    #nsampling = int(sys.argv[4])

    viscous_on = "true"
    force_decay = "true"
    nsampling = 1

    for eid in range(200):
        #try:
        fpath_event = os.path.join(fpath, "event%s"%eid)
        main(fpath_event, viscous_on, force_decay, nsampling=nsampling)
        #except:
        #    print("event%s is not exist"%eid)


