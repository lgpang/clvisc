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

import pyopencl as cl
from pyopencl.array import Array
import pyopencl.array as cl_array


os.environ['PYOPENCL_CTX']=':0'

class Polarization(object):
    '''The pyopencl version for lambda polarisation,
    initialize with freeze out hyper surface and omega^{mu}
    on freeze out hyper surface.'''
    def __init__(self, sf, omega):
        '''Param:
             sf: the freeze out hypersf ds0,ds1,ds2,ds3,vx,vy,veta,etas
             omega: omega^tau, x, y, etas
        '''
        self.cwd, cwf = os.path.split(__file__)
        self.mass = 1.115
        self.Tfrz = 0.137
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        src = open('kernel_polarization.cl', 'r').read()
        self.prg = cl.Program(self.ctx, src).build()

        # calc umu since they are used for each (Y,pt,phi)
        vx = sf[:, 4]
        vy = sf[:, 5]
        vz = sf[:, 6]
        v_sqr = vx*vx + vy*vy + vz*vz
        v_sqr[v_sqr>1.0] = 0.99999
        u0 = 1.0/np.sqrt(1.0 - v_sqr)

        self.size_sf = len(sf[:,0])

        h_umu = np.zeros((self.size_sf, 4), dtype=np.float32)
        h_umu[:, 0] = u0
        h_umu[:, 1] = u0 * vx
        h_umu[:, 2] = u0 * vy
        h_umu[:, 3] = u0 * vz

        h_smu = sf[:, 0:4].astype(np.float32)
        h_etas = sf[:, 7].astype(np.float32)
        h_omegaY = 0.5*omega[:, 2].astype(np.float32)

        mf = cl.mem_flags
        self.d_smu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_smu)
        self.d_umu = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_umu)
        self.d_omegaY = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omegaY)
        self.d_etas = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_etas)

        self.d_pol = Array(self.queue, self.size_sf, np.float32)
        self.d_vor = Array(self.queue, self.size_sf, np.float32)
        self.d_rho = Array(self.queue, self.size_sf, np.float32)

    def pol_vor_rho(self, Y, px, py):
        '''return: polarization, vorticity, density at given Y,px,py'''
        self.prg.polarization_on_sf(self.queue, (self.size_sf,), None,
            self.d_pol.data, self.d_vor.data, self.d_rho.data,
            self.d_smu, self.d_umu, self.d_omegaY, self.d_etas,
            np.float32(Y), np.float32(px), np.float32(py),
            np.int32(self.size_sf)).wait()

        polarization = cl_array.sum(self.d_pol).get()
        vorticity = cl_array.sum(self.d_vor).get()
        density = cl_array.sum(self.d_rho).get()
        return polarization, vorticity, density


