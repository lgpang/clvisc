#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 25 Feb 2016 14:13:27 CET

'''calc the Lambda polarization on the freeze out hypersurface'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import sys
from time import time
import math
import h5py

import four_momentum as mom
from numba import jit
import matplotlib.pyplot as plt

from common_plotting import smash_style

# store the data in hdf5 file
f_h5 = h5py.File('vor.hdf5', 'w')

dset_pt = f_h5.create_dataset('mom/PT', (mom.NPT, ), dtype='f')
dset_pt[...] = mom.PT
dset_phi = f_h5.create_dataset('mom/PHI', (mom.NPHI, ), dtype='f')
dset_phi[...] = mom.PHI
rapidity = np.linspace(-5, 5, 11, endpoint=True)
dset_rapidity = f_h5.create_dataset('mom/Y', (len(rapidity), ), dtype='f')
dset_rapidity[...] = rapidity


class LambdaPolarisation(object):
    def __init__(self, sf, omega, fpath='./', event_id=0):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        self.cwd, cwf = os.path.split(__file__)
        self.mass = 1.115
        self.Tfrz = 0.137
        self.sf = sf
        self.omega = omega
        self.fpath = fpath
        self.event_id = event_id

    @jit
    def vorticity_int(self, fpath='./'):
        ''' The pt and phi integrated vorticity at different rapidities '''
        npt, nphi = mom.NPT, mom.NPHI
        vor = np.zeros((npt, nphi))
        rho = np.zeros((npt, nphi))

        vor_int = []
        rho_int = []

        for Y in rapidity:
            for i, pt in enumerate(mom.PT):
                for j, phi in enumerate(mom.PHI):
                    px = pt * math.cos(phi)
                    py = pt * math.sin(phi)
                    pol_ij, omg_ij, rho_ij = self.Pimu_rho(Y, px, py)
                    vor[i, j] = pol_ij
                    rho[i, j] = rho_ij
            dset_vor = f_h5.create_dataset('event%s/vor_vs_pt_phi/rapidity%s'%(self.event_id, Y),
                    (npt, nphi), dtype='f')
            dset_vor[...] = vor
            dset_rho = f_h5.create_dataset('event%s/rho_vs_pt_phi/rapidity%s'%(self.event_id, Y),
                    (npt, nphi), dtype='f')
            dset_rho[...] = rho

            vor_int.append( mom.pt_phi_integral(vor) )
            rho_int.append( mom.pt_phi_integral(rho) )

            print(Y, 'finished')

        dset_vorint = f_h5.create_dataset('event%s/integral_pt_phi/vor'%self.event_id, (len(rapidity), ), dtype='f')
        dset_vorint[...] = np.array(vor_int)
        dset_rhoint = f_h5.create_dataset('event%s/integral_pt_phi/rho'%self.event_id, (len(rapidity), ), dtype='f')
        dset_rhoint[...] = np.array(rho_int)

    def vorticity_vs_pxpy(self, Y=4):
        nx, ny = 20, 20
        vor = np.zeros((nx, ny))
        omg = np.zeros((nx, ny))
        for ix, px in enumerate(np.linspace(-3, 3, nx)):
            for iy, py in enumerate(np.linspace(-3, 3, ny)):
                vor_y, omega_y, rho = self.Pimu_rho(Y, px, py)
                vor[ix, iy] = vor_y/rho
                omg[ix, iy] = omega_y/rho
            print(px, 'finished')

        #plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3), vmin=-0.01, vmax=0.01)
        np.savetxt('%s/pol_Y%s.dat'%(self.fpath, Y), vor)

        vmax = vor.max()
        vmin = vor.min()
        if vmax < -vmin:
            vmax = -vmin
        plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3), vmin=-vmax, vmax=vmax)
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\Pi^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('%s/vor_Y%s.png'%(self.fpath, Y))
        plt.close()

        vmax = omg.max()
        vmin = omg.min()
        if vmax < -vmin:
            vmax = -vmin
        plt.imshow(omg.T, extent=(-3,3,-3,3), cmap=plt.get_cmap('bwr'), origin='lower', vmin=-vmax, vmax=vmax)
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\omega^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('%s/omg_Y%s.png'%(self.fpath, Y))
        plt.close()

    @jit
    def Pimu_rho(self, Y, px, py):
        '''give the polarization along y and total density'''
        mass = self.mass
        Tfrz = self.Tfrz
        omega_y = 0.5*self.omega[:, 2]

        ds0 = self.sf[:, 0]
        ds1 = self.sf[:, 1]
        ds2 = self.sf[:, 2]
        ds3 = self.sf[:, 3]
        vx = self.sf[:, 4]
        vy = self.sf[:, 5]
        vz = self.sf[:, 6]
        etas = self.sf[:, 7]

        pt = math.sqrt(px*px + py*py)
        mt = math.sqrt(mass*mass + pt*pt)
        mtcy = mt * np.cosh(Y - etas)
        mtsy = mt * np.sinh(Y - etas)

        v_sqr = vx*vx + vy*vy + vz*vz
        v_sqr[v_sqr>1.0] = 0.99999
        u0 = 1.0/np.sqrt(1.0 - v_sqr)

        pdotu = u0*(mtcy-px*vx-py*vy-mtsy*vz)
        volum = mtcy*ds0 - px*ds1 - py*ds2 - mtsy*ds3

        #nf = 1.0/(np.exp(pdotu/Tfrz) + 1.0)
        tmp = np.exp(-pdotu/Tfrz)
        nf = tmp/(1.0 + tmp)

        pbar_sqr = mass*mass - pdotu*pdotu

        beta = 1.0/Tfrz

        mass_fkt = 1.0 - pbar_sqr/(3*mass*(mass+pdotu))

        # n \cdot omega = n0*omega0 - nx*omega_x - ny*omega_y - nz*omega_z
        # n = (0, 1, 0, 0)
        total_polarization = -(volum*beta*omega_y*mass_fkt*pbar_sqr/(pdotu*pdotu)*nf*(1-nf)).sum()/6.0

        total_omega = (volum*omega_y*nf).sum()

        total_density = (volum * nf).sum()

        return total_polarization, total_omega, total_density



for eid in range(50):
    fpath = '/tmp/lgpang/vorticity/cent30_35_event%s_mod/'%eid
    sf = np.loadtxt('%s/hypersf.dat'%fpath)
    omega = np.loadtxt('%s/omegamu_sf.dat'%fpath)

    polar = LambdaPolarisation(sf, omega, fpath, eid)
    polar.vorticity_int()

#vor = []
#for Y in range(-6, 7):
#    vor_int_px = polar.vorticity(Y)
#    vor.append(vor_int_px)

