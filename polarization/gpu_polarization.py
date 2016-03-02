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


#os.env['PYOPENCL_CTX']=':2'
class Polarization(object):
    '''The pyopencl version for lambda polarisation'''
    def __init__(self, sf, omega):
        self.cwd, cwf = os.path.split(__file__)
        self.mass = 1.115
        self.Tfrz = 0.137
        self.sf = sf
        self.omega = omega
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        src = open('kernel_polarization.cl', 'r').read()
        self.prg = cl.Program(self.ctx, src).build()

        vx = self.sf[:, 4]
        vy = self.sf[:, 5]
        vz = self.sf[:, 6]
        v_sqr = vx*vx + vy*vy + vz*vz
        v_sqr[v_sqr>1.0] = 0.99999
        u0 = 1.0/np.sqrt(1.0 - v_sqr)

        self.size_sf = len(sf[:,0])
        print(self.size_sf)

        h_umu = np.zeros((self.size_sf, 4), dtype=np.float32)
        h_umu[:, 0] = u0
        h_umu[:, 1] = u0 * vx
        h_umu[:, 2] = u0 * vy
        h_umu[:, 3] = u0 * vz

        h_umu = h_umu.astype(np.float32)
        h_smu = sf[:, 0:4].astype(np.float32)
        h_etas = sf[:, 7].astype(np.float32)
        omega = omega.astype(np.float32)

        mf = cl.mem_flags
        self.d_smu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_smu)
        self.d_umu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_umu)
        self.d_omegaY = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omega)
        self.d_etas = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_etas)

        self.h_pol = np.zeros_like(omega)
        self.h_vor = np.zeros_like(omega)
        self.h_rho = np.zeros_like(omega)

        self.d_pol = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_pol)
        self.d_vor = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_vor)
        self.d_rho = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.h_rho)

        from pyopencl.reduction import ReductionKernel
        self.sum_on_gpu = ReductionKernel(self.ctx, np.float32, neutral="0",
                        reduce_expr="a+b", map_expr="x[i]", arguments="__global float *x")

    def pol_vor_rho(self, Y, px, py):
        '''return: polarization, vorticity, density '''
        self.prg.polarization_on_sf(self.queue, (self.size_sf,), None,
            self.d_pol, self.d_vor, self.d_rho,
            self.d_smu, self.d_umu, self.d_omegaY, self.d_etas,
            np.float32(Y), np.float32(px), np.float32(py),
            np.int32(self.size_sf)).wait()

        cl.enqueue_copy(self.queue, self.h_pol, self.d_pol).wait()
        cl.enqueue_copy(self.queue, self.h_vor, self.d_vor).wait()
        cl.enqueue_copy(self.queue, self.h_rho, self.d_rho).wait()
        polarization = self.h_pol.sum()
        vorticity = self.h_vor.sum()
        density = self.h_rho.sum()
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

        print('rapidity', Y, 'finished')

        return vor.sum()*(6.0/nx)**2.0


from numpy import genfromtxt

fpath = '/tmp/lgpang/vorticity/cent30_35_event2_mod/'

sf = np.loadtxt('%s/hypersf.dat'%fpath, dtype=np.float32)
omega = np.loadtxt('%s/omegamu_sf.dat'%fpath, dtype=np.float32)

fpath = './'
polar = Polarization(sf, omega)
#polar.pol_vor_rho(0, 2, 0)

vor = []
for Y in range(-6, 7):
    vor_int_px = polar.vorticity(Y)
    vor.append(vor_int_px)

print(vor)


