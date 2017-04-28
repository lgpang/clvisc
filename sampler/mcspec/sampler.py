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

def get_ptspec(fpath, dat, pid=211, nsampling=2000, kind='Y', rapidity_window=1.6):
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

    pti = pti[np.abs(Yi)<0.5*rapidity_window]

    dN, pt = np.histogram(pti, bins=50)

    dpt = pt[1:]-pt[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])

    res = np.array([pt, dN/(2*np.pi*float(nsampling)*pt*dpt*rapidity_window)]).T

    fname = os.path.join(fpath, 'dN_over_2pid%sptdpt_mc_%s.dat'%(kind, pid))
    np.savetxt(fname, res)
    #return res[:, 0], res[:, 1]

def plot(fpath, particle_lists, nsampling):
    Y0, dNdY_charged = get_dNdY(fpath, particle_lists, pid='charged', nsampling=nsampling, kind='Eta')
    Y0, dNdY_pion = get_dNdY(fpath, particle_lists, pid=211, nsampling=nsampling, kind='Eta')
    Y0, dNdY_kaon = get_dNdY(fpath, particle_lists, pid=321, nsampling=nsampling, kind='Eta')
    Y0, dNdY_proton = get_dNdY(fpath, particle_lists, pid=2212, nsampling=nsampling, kind='Eta')

    get_ptspec(fpath, particle_lists, pid=211,  nsampling=nsampling, kind='Y', rapidity_window=1.6)
    get_ptspec(fpath, particle_lists, pid=321,  nsampling=nsampling, kind='Y', rapidity_window=1.6)
    get_ptspec(fpath, particle_lists, pid=2212, nsampling=nsampling, kind='Y', rapidity_window=1.6)
    get_ptspec(fpath, particle_lists, pid='charged', nsampling=nsampling,  kind='Eta', rapidity_window=1.6)



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

    print('particle list read in')

    #np.savetxt('mc_particle_list.txt', particle_lists)

    print('particle list saved')

    from mcspec import mcspec

    #mcspec(fstring(proc))

    os.chdir(cwd)

    plot(fpath, particle_lists, nsampling = nsampling)



if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        print('usage:python sampler.py fpath viscous_on  force_decay nsampling')
        exit(0)

    fpath = sys.argv[1]
    viscous_on = sys.argv[2]
    force_decay = sys.argv[3]
    nsampling = int(sys.argv[4])
    main(fpath, viscous_on, force_decay, nsampling=nsampling)


