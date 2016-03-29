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
import four_momentum as mom

import pyopencl as cl
from pyopencl.array import Array
import pyopencl.array as cl_array


os.environ['PYOPENCL_CTX']=':1'

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
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        # calc umu since they are used for each (Y,pt,phi)
        self.size_sf = len(sf[:,0])

        h_sf = sf.astype(np.float32)
        h_omega = 0.5*omega.astype(np.float32)
        print(h_omega)

        mf = cl.mem_flags
        self.d_sf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_sf)
        self.d_omega = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega)


    def pol_rho(self, Y, px, py):
        '''return: polarization, density at given Y,px,py list'''
        d_Y = cl_array.to_device(self.queue, Y.astype(np.float32))
        d_px = cl_array.to_device(self.queue, px.astype(np.float32))
        d_py = cl_array.to_device(self.queue, py.astype(np.float32))

        nY, npx, npy = len(Y), len(px), len(py)
        compile_options = ['-D NRAPIDITY=%s'%nY, '-D NPX=%s'%npx, '-D NPY=%s'%npy]

        cwd, cwf = os.path.split(__file__)
        compile_options.append('-I '+os.path.join(cwd, '../kernel/')) 
        compile_options.append( '-D USE_SINGLE_PRECISION' )

        fpath = os.path.join(cwd, '../kernel/kernel_polarization.cl')

        with open(fpath, 'r') as f:
            src = f.read()
            self.prg = cl.Program(self.ctx, src).build(options=compile_options)

        size = nY * npx * npy
        h_pol = np.zeros((size, 4), np.float32)
        h_rho = np.zeros(size, np.float32)
        self.d_pol = cl_array.to_device(self.queue, h_pol)
        self.d_rho = cl_array.to_device(self.queue, h_rho)

        self.prg.polarization_on_sf(self.queue, (256,), (256,),
            self.d_pol.data, self.d_rho.data, self.d_sf, d_Y.data,
            self.d_omega, d_Y.data, d_px.data, d_py.data,
            np.int32(self.size_sf)).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        return polarization, density


if __name__ == '__main__':
    #sf = np.loadtxt('../for_polarization_test/hypersf.dat', dtype=np.float32)
    #omega = np.loadtxt('../for_polarization_test/omegamu_sf.dat', dtype=np.float32).flatten()
    fpath = '/lustre/nyx/hyihp/lpang/auau200_results/cent20_30/etas0p08/event0/'
    sf = np.loadtxt(fpath+'/hypersf.dat', dtype=np.float32)
    omega = np.loadtxt(fpath+'/omegamu_sf.dat', dtype=np.float32).flatten()
    pol = Polarization(sf, omega)

    Y = np.linspace(-5, 5, 11, endpoint=True)
    px = np.linspace(-3, 3, 21, endpoint=True)
    py = np.linspace(-3, 3, 21, endpoint=True)
    polar, density = pol.pol_rho(Y, px, py)
    np.savetxt('polar.dat', polar)
    np.savetxt('density.dat', density)


