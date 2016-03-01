#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 25 Feb 2016 14:13:27 CET

'''calc the Lambda polarization on the freeze out hypersurface'''

from __future__ import absolute_import, division, print_function
#from pyopencl import array
import numpy as np
import os
import sys
from time import time
import math

import four_momentum as mom
from numba import jit
import matplotlib.pyplot as plt

from common_plotting import smash_style

import pyopencl as cl
from pyopencl.array import Array
import pyopencl.array as cl_array

class Polarization(object):
    '''The pyopencl version for lambda polarisation'''
    src = '''
    # define mass 1.115f
    # define beta (1.0f/0.137f)

    __kernel void polarization_on_sf(
                __global float * d_polarization,
                __global float * d_vorticity,
                __global float * d_density,
                __global const float4 * d_s,
                __global const float4 * d_u,
                __global const float * d_omegaY,
                __global const float * d_etas,
                __const float rapidity,
                __const float px,
                __const float py) {
                    int i = get_global_id(0);
                    float4 dsmu = d_s[i];
                    float4 umu = d_u[i];
                    float omega_y = d_omegaY[i];
                    float etas = d_etas[i];

                    float pt = sqrt(px*px + py*py);
                    float mt = sqrt(mass*mass + pt*pt);
                    float4 p_mu = (float4)(mt * cosh(rapidity - etas),
                                -px, -py, -mt * sinh(rapidity - etas));
                    float pdotu = dot(umu, p_mu);

                    float volum = dot(dsmu, p_mu);
            
                    float tmp = exp(-pdotu*beta);
                    float nf = tmp/(1.0f + tmp);
            
                    float pbar_sqr = mass*mass - pdotu*pdotu;
            
                    float mass_fkt = 1.0f - pbar_sqr/(3*mass*(mass+pdotu));
            
                    d_polarization[i] = -(volum*beta*omega_y*mass_fkt*pbar_sqr/(pdotu*pdotu)*nf*(1-nf))/6.0f;
            
                    d_vorticity[i] = volum*omega_y*nf;
            
                    d_density[i] = volum * nf;
                }
    '''
    def __init__(self, sf, omega):
        self.cwd, cwf = os.path.split(__file__)
        self.mass = 1.115
        self.Tfrz = 0.137
        self.sf = sf
        self.omega = omega
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, self.src).build()

        vx = self.sf[:, 4]
        vy = self.sf[:, 5]
        vz = self.sf[:, 6]
        v_sqr = vx*vx + vy*vy + vz*vz
        v_sqr[v_sqr>1.0] = 0.99999
        u0 = 1.0/np.sqrt(1.0 - v_sqr)

        self.size_sf = len(sf[:,0])
        h_umu = np.zeros((self.size_sf, 4))
        h_umu[:, 0] = u0
        h_umu[:, 1] = u0 * vx
        h_umu[:, 2] = u0 * vy
        h_umu[:, 3] = u0 * vz

        mf = cl.mem_flags
        self.d_smu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sf[:,0:4])
        self.d_umu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_umu)
        self.d_omegaY = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omega)
        self.d_etas = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sf[:,7])

        self.d_pol = Array(self.queue, (self.size_sf, 1), np.float32)
        self.d_vor = Array(self.queue, (self.size_sf, 1), np.float32)
        self.d_rho = Array(self.queue, (self.size_sf, 1), np.float32)

        from pyopencl.reduction import ReductionKernel
        self.sum_on_gpu = ReductionKernel(self.ctx, np.float32, neutral="0",
                        reduce_expr="a+b", map_expr="x[i]", arguments="__global float *x")

    def pol_vor_rho(self, Y, px, py):
        '''return: polarization, vorticity, density '''
        self.prg.polarization_on_sf(self.queue, (self.size_sf,), None,
            self.d_pol.data, self.d_vor.data, self.d_rho.data,
            self.d_smu, self.d_umu, self.d_omegaY, self.d_etas,
            np.float32(Y), np.float32(px), np.float32(py)).wait()

        #polarization = self.sum_on_gpu(self.d_pol).get()
        polarization = cl_array.sum(self.d_pol)
        vorticity = cl_array.sum(self.d_vor)
        density = cl_array.sum(self.d_rho)
        print(polarization, vorticity, density)

        return polarization, vorticity, density

    def vorticity(self, Y=4):
        nx, ny = 20, 20
        vor = np.zeros((nx, ny))
        omg = np.zeros((nx, ny))
        for ix, px in enumerate(np.linspace(-3, 3, nx)):
            for iy, py in enumerate(np.linspace(-3, 3, ny)):
                vor_y, omega_y, rho = self.pol_vor_rho(Y, px, py)
                vor[ix, iy] = vor_y/rho
                omg[ix, iy] = omega_y/rho
            print(px, 'finished')

        #plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3), vmin=-0.01, vmax=0.01)
        plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3))
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\Pi^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('vor_Y%s.png'%Y)
        plt.close()

        plt.imshow(omg.T, extent=(-3,3,-3,3), cmap=plt.get_cmap('bwr'), origin='lower', vmin=-0.01, vmax=0.01)
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\omega^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('omg_Y%s.png'%Y)
        plt.close()







class LambdaPolarisation(object):
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
        omg = np.zeros((nx, ny))
        for ix, px in enumerate(np.linspace(-3, 3, nx)):
            for iy, py in enumerate(np.linspace(-3, 3, ny)):
                vor_y, omega_y, rho = self.Pimu_rho(Y, px, py)
                vor[ix, iy] = vor_y/rho
                omg[ix, iy] = omega_y/rho
            print(px, 'finished')

        #plt.imshow(vor.T, cmap=plt.get_cmap('bwr'), origin='lower', extent=(-3,3,-3,3), vmin=-0.01, vmax=0.01)
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
        plt.savefig('vor_Y%s.png'%Y)
        plt.close()

        plt.imshow(omg.T, extent=(-3,3,-3,3), cmap=plt.get_cmap('bwr'), origin='lower', vmin=-0.01, vmax=0.01)
        plt.xlabel(r'$p_x\ [GeV]$')
        plt.ylabel(r'$p_y\ [GeV]$')
        plt.title(r'$\omega^{y}\ @\ rapidity=%s$'%Y)
        plt.colorbar()
        smash_style.set()
        plt.savefig('omg_Y%s.png'%Y)
        plt.close()

        return vor.sum()*(6.0/nx)**2.0

    @jit
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

        pbar_sqr = mass*mass - pdotu*pdotu

        beta = 1.0/Tfrz

        mass_fkt = 1.0 - pbar_sqr/(3*mass*(mass+pdotu))

        # n \cdot omega = n0*omega0 - nx*omega_x - ny*omega_y - nz*omega_z
        # n = (0, 1, 0, 0)
        total_polarization = -(volum*beta*omega_y*mass_fkt*pbar_sqr/(pdotu*pdotu)*nf*(1-nf)).sum()/6.0

        total_omega = (volum*omega_y*nf).sum()

        total_density = (volum * nf).sum()

        return total_polarization, total_omega, total_density



from numpy import genfromtxt

sf = np.loadtxt('hypersf.dat')
omega = np.loadtxt('omegamu_sf.dat')

polar = LambdaPolarisation(sf, omega)

gpu_polar = Polarization(sf, omega)

vor = []
for Y in range(-6, 7):
    vor_int_px = polar.vorticity(Y)
    vor.append(vor)

rapidity = np.linspace(-6, 6, 13, endpoint=True)
plt.plot(rapidity, vor)
plt.xlabel(r'$rapidity$')
plt.ylabel(r'$\int P^y dp_x dp_y$')
smash_style.set()
plt.savefig('Pi_int_pxpy.png')
plt.close()
