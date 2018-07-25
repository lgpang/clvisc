#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 May 2015 17:51:18 CEST
import numpy as np
import sympy as sym
import pandas as pd
import os
from subprocess import call, check_output
import cmath

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

    particle_type = None

    if pid == 'charged':
        particle_type = (dat[:, 5]==dat[:, 5])
    else:
        particle_type = (dat[:, 5]==pid)

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

def get_event_planes(fpath, dat, pid=211, nsampling=2000, kind='Y',Ylo=3.3, Yhi=3.9, total_n=6,
                     num_ptbins=15, num_phibins=50):
    rapidity_col = 4
    if kind == 'Eta':
        rapidity_col = 6

    particle_type = None
    if pid == 'charged':
        particle_type = (dat[:, 5]==dat[:, 5])
    else:
        particle_type = (dat[:, 5]==pid)

    Yi = dat[particle_type, rapidity_col]
    phi_p = np.arctan2(dat[particle_type,2], dat[particle_type,1])
    pti = np.sqrt(dat[particle_type, 1]**2 + dat[particle_type, 2]**2) 

    pti = pti[(Yi>Ylo)*(Yi<Yhi)]
    phi_p = phi_p[(Yi>Ylo)*(Yi<Yhi)] 
    d2N, pt, Phi = np.histogram2d(pti, phi_p, range=[[0, 4.0], [-np.pi, np.pi]], bins=[num_ptbins, num_phibins]) 
    dpt = pt[1:]-pt[:-1] 
    dphi_p = Phi[1:]-Phi[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])
    Phi = 0.5*(Phi[1:]+Phi[:-1]) 
    
    d2N=d2N.flatten()  
    Phi = np.repeat(Phi, num_ptbins)
    
    Vn = np.zeros(total_n+1)
    event_plane = np.zeros(total_n+1)
    total_vn = np.zeros(total_n+1, dtype=complex)
    Norm = np.sum(d2N)
    print('event_plane_window',Norm,pid)
    for n in range(1, total_n+1):
        total_vn[n] = (d2N * np.exp(1j*n*Phi)).sum()/float(Norm)
        Vn[n], event_plane[n]=cmath.polar(total_vn[n])
        event_plane[n] /= float(n)
    return event_plane

def get_vn_pt(fpath, dat, pid=211, nsampling=2000, kind='Y',Ylo=-0.35, Yhi=0.35,
        Y_event_plane=[3.3, 3.9], total_n=6, num_ptbins=15, num_phibins=50):
    rapidity_col = 4
    if kind == 'Eta':
        rapidity_col = 6
    particle_type = None 
    if pid == 'charged':
        particle_type = (dat[:, 5]==dat[:, 5])
    else:
        particle_type = (dat[:, 5]==pid)

    Yi = dat[particle_type, rapidity_col]
    phi_p = np.arctan2(dat[particle_type,2], dat[particle_type,1])
    pti = np.sqrt(dat[particle_type, 1]**2 + dat[particle_type, 2]**2)
    pti = pti[np.abs(Yi)<Yhi]
    phi_p = phi_p[np.abs(Yi)<Yhi]
    d2N, pt, Phi = np.histogram2d(pti, phi_p, range=[[0, 4.0], [-np.pi, np.pi]], bins=[num_ptbins, num_phibins])
    dpt = pt[1:]-pt[:-1]
    dphi_p = Phi[1:]-Phi[:-1]
    pt = 0.5*(pt[1:]+pt[:-1])
    Phi = 0.5*(Phi[1:]+Phi[:-1])
    
    Vn_pt = np.zeros(shape=(num_ptbins, total_n+1))
    Vn_vec = np.zeros(shape=(num_ptbins, total_n+1), dtype=complex)
    angles = np.zeros(shape=(num_ptbins, total_n+1))
    event_plane = np.zeros(total_n+1)
    Norm = 0
    event_plane = get_event_planes(fpath,dat,pid,nsampling,kind, Ylo=Y_event_plane[0], Yhi=Y_event_plane[1])
    for i in range(num_ptbins):
        norm_factor = np.sum(d2N[i,:])
        if norm_factor<1e-2: continue
        print('norm,pti',norm_factor,pt[i])
        for n in range(1, 7):
            Vn_vec[i, n] = (d2N[i]*np.exp(1j*n*(Phi-event_plane[n]))).sum()/float(norm_factor)
            Vn_pt[i, n], angles[i, n]=cmath.polar(Vn_vec[i, n])
    fout_name = os.path.join(fpath,'vn_mc_%s.dat'%pid)
    np.savetxt(fout_name, np.array(list(zip(pt, Vn_pt[:,1],Vn_pt[:,2],Vn_pt[:,3],Vn_pt[:,4],Vn_pt[:,5], Vn_pt[:,6]))))



def plot(fpath, particle_lists, nsampling):
    Y0, dNdY_charged = get_dNdY(fpath, particle_lists, pid='charged', nsampling=nsampling, kind='Eta')
    get_ptspec(fpath, particle_lists, pid=211,  nsampling=nsampling, kind='Y', rapidity_window=1.0)
    get_ptspec(fpath, particle_lists, pid=321,  nsampling=nsampling, kind='Y', rapidity_window=1.0)
    get_ptspec(fpath, particle_lists, pid=2212, nsampling=nsampling, kind='Y', rapidity_window=1.0)
    get_ptspec(fpath, particle_lists, pid='charged', nsampling=nsampling,  kind='Eta', rapidity_window=1.6)

    get_vn_pt(fpath, particle_lists, pid=211, nsampling=nsampling, kind='Y', Ylo=-0.5, Yhi=0.5)
    get_vn_pt(fpath, particle_lists, pid=321, nsampling=nsampling, kind='Y', Ylo=-0.5, Yhi=0.5)
    get_vn_pt(fpath, particle_lists, pid=2212, nsampling=nsampling, kind='Y', Ylo=-0.5, Yhi=0.5)
    get_vn_pt(fpath, particle_lists, pid='charged', nsampling=nsampling, kind='Y', Ylo=-0.5, Yhi=0.5)
    



def main(fpath, viscous_on, force_decay, nsampling):
    cwd = os.getcwd()
    os.chdir('../build')
    #call(['cmake', '..'])
    #call(['make'])

    ns_str = '%s'%nsampling
    cmd = ['./main', fpath, viscous_on, force_decay, ns_str]

    proc = check_output(cmd)

    fstr = None
    try:
        # used in python 2.*
        from StringIO import StringIO as fstring
        fstr = fstring(str(proc))
    except ImportError:
        # used in python 3.*
        #from io import BytesIO as fstring
        from io import StringIO as fstring
        fstr = fstring(str(proc, 'utf-8'))


    particle_lists = pd.read_csv(fstr, sep=' ', header=None,
            dtype=np.float32, comment='#').values

    #print('particle list read in')
    #np.savetxt('mc_particle_list.txt', particle_lists)
    #print('particle list saved')

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
    fsrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdg05.dat")
    call(['cp', fsrc, fpath])

    main(fpath, viscous_on, force_decay, nsampling=nsampling)


