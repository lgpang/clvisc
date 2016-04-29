#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import pandas as pd
from common_plotting import smash_style
import os

def get_dNdY(fpath, dat, pid=211, nsampling=2000, kind='Y'):
    rapidity_col = 4
    if kind == 'Eta':
        rapidity_col = 6
    Yi = dat[:, rapidity_col]

    dN, Y = None, None
    if pid == 'charged':
        dN, Y = np.histogram(Yi, bins=50)
    else:
        dN, Y = np.histogram(Yi[dat[:, 5]==pid], bins=50)
    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])
    res = np.array([Y, dN/(dY*float(nsampling))]).T
    np.savetxt(os.path.join(fpath, 'dNd%s_mc_%s.dat'%(kind, pid)), res)
    return res[:, 0], res[:, 1]

def get_ptspec(fpath, dat, pid=211, nsampling=2000, kind='Y'):
    E = dat[:,0]
    pz = dat[:,3]
    rapidity_col = 4
    if kind == 'Eta':
        rapidity_col = 6

    particle_type = (dat[:, 5]==pid)

    if pid == 'charged':
        particle_type = (dat[:, 5]==dat[:, 5])

    Yi = dat[particle_type, rapidity_col]

    dN, Y = np.histogram(Yi, bins=50)

    dY = (Y[1:]-Y[:-1])
    Y = 0.5*(Y[:-1]+Y[1:])

    pti = np.sqrt(dat[particle_type, 1]**2+dat[particle_type, 2]**2)

    pti = pti[np.abs(Yi)<0.8]

    dN, pt = np.histogram(pti, bins=50)

    dpt = pt[1:]-pt[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])

    res = np.array([pt, dN/(2*np.pi*float(nsampling)*pt*dpt*1.6)]).T

    fname = os.path.join(fpath, 'dN_over_2pid%sptdpt_mc_%s.dat'%(kind, pid))
    np.savetxt(fname, res)
    #return res[:, 0], res[:, 1]

#@profile
def plot(fpath, particle_lists, kind):

    Y0, dNdY_charged = get_dNdY(fpath, particle_lists, pid='charged', kind=kind)

    Y0, dNdY_nodecay = get_dNdY(fpath, particle_lists, pid=211, kind=kind)

    Y2_kaon, dNdY_kaon = get_dNdY(fpath, particle_lists, pid=321, kind=kind)

    Y2_proton, dNdY_proton = get_dNdY(fpath, particle_lists, pid=2212, kind=kind)

    get_ptspec(fpath, particle_lists, pid=211, kind=kind)
    get_ptspec(fpath, particle_lists, pid=321, kind=kind)
    get_ptspec(fpath, particle_lists, pid=2212, kind=kind)
    get_ptspec(fpath, particle_lists, pid='charged', kind=kind)



def main(fpath, viscous_on, force_decay):
    from subprocess import call, check_output
    cwd = os.getcwd()
    os.chdir('../build')
    call(['cmake', '..'])
    call(['make'])

    cmd = ['./main', fpath, viscous_on, force_decay]

    proc = check_output(cmd)

    try:
        # used in python 2.*
        from StringIO import StringIO as fstring
    except ImportError:
        # used in python 3.*
        from io import StringIO as fstring

    particle_lists = np.genfromtxt(fstring(proc))

    os.chdir(cwd)

    plot(fpath, particle_lists, kind='Eta')
    plot(fpath, particle_lists, kind='Y')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        print('usage:python dNdY_test.py fpath viscous_on  force_decay')
        exit(0)

    fpath = sys.argv[1]
    viscous_on = sys.argv[2]
    force_decay = sys.argv[3]
    main(fpath, viscous_on, force_decay)


