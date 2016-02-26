#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 25 Feb 2016 14:13:27 CET

'''calc the Lambda polarization on the freeze out hypersurface'''

from __future__ import absolute_import, division, print_function
#from pyopencl import array
import numpy as np
#import pyopencl as cl
import os
import sys
from time import time
import math

import four_momentum as mom
from numba import jit
import matplotlib.pyplot as plt

from common_plotting import smash_style


class LambdaPolarisation(object):
    '''The pyopencl version for lambda polarisation'''
    def __init__(self, sf, omega):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        self.cwd, cwf = os.path.split(__file__)
        self.mass = 1.115
        self.Tfrz = 0.137
        self.sf = sf
        self.omega = omega


    def vorticity(self, Y=4):
        nx, ny = 20, 20
        vor = np.zeros((nx, ny))
        for ix, px in enumerate(np.linspace(-3, 3, nx)):
            for iy, py in enumerate(np.linspace(-3, 3, ny)):
                vor_y, rho = self.Pimu_rho(Y, px, py)
                vor[ix, iy] = vor_y/rho
            print(px, 'finished')

        plt.imshow(vor.T, extent=(-3,3,-3,3))
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\Pi^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('vor_Y%s.png'%Y)
        plt.close()
        np.savetxt('vor_Y%s.dat'%Y, vor, header='x=[-3,3], y=[-3, 3], 20*20 grids')

    #@jit
    def Pimu_rho(self, Y, px, py):
        '''give the polarization along y and total density'''
        mass = self.mass
        Tfrz = self.Tfrz
        omega_y = self.omega[:, 2]

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

        pbar_sqr = px*px + py*py + mtsy*mtsy

        beta = 1.0/Tfrz

        mass_fkt = 1.0 - pbar_sqr/(3*mass*(mass+pdotu))

        # n \cdot omega = n0*omega0 - nx*omega_x - ny*omega_y - nz*omega_z
        # n = (0, 1, 0, 0)
        total_polarization = -(volum*beta*omega_y*mass_fkt*pbar_sqr/(pdotu*pdotu)*nf*(1-nf)).sum()/6.0

        total_density = (volum * nf).sum()

        return total_polarization, total_density



from numpy import genfromtxt
#sf = np.loadtxt('../results/P30_wbmod_etaos0/hypersf.dat')
#omega = np.loadtxt('../results/P30_wbmod_etaos0/omegamu_sf.dat')
#sf = np.genfromtxt('../results/P3_wbmod_etaos0p08/hypersf.dat', filling_values=0.0)
#omega = np.genfromtxt('../results/P3_wbmod_etaos0p08/omegamu_sf.dat', filling_values=0.0)

sf = np.loadtxt('../results/P30_EOSI_etaos0_Tfrz0p02/hypersf.dat')
omega = np.loadtxt('../results/P30_EOSI_etaos0_Tfrz0p02/omegamu_sf.dat')

polar = LambdaPolarisation(sf, omega)

for Y in range(-6, 7):
    polar.vorticity(Y)
