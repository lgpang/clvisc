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
        h_omega = omega.astype(np.float32)
        print(h_omega)

        mf = cl.mem_flags
        self.d_sf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_sf)
        self.d_omega = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega)


    def get(self, momentum_list, tilte=1):
        '''return: polarization, density at given momentum
        Params:
            :param momentum: a numpy array with shape (size, 4)
                where size= number of different four-momentum vector
                in each vector, p4=(mt, Y, px, py)
            :param tilte: num of calculations in each workgroup '''
        d_momentum = cl_array.to_device(self.queue,
                                        momentum_list.astype(np.float32))

        num_of_mom = len(momentum_list)

        print('num_of_mom=', num_of_mom)

        compile_options = ['-D num_of_mom=%s'%num_of_mom, '-D title=%s'%tilte]

        cwd, cwf = os.path.split(__file__)

        block_size = 256
        compile_options = ['-D BSZ=%s'%block_size]
        compile_options.append('-I '+os.path.join(cwd, '../kernel/')) 
        compile_options.append( '-D USE_SINGLE_PRECISION' )

        fpath = os.path.join(cwd, '../kernel/kernel_polarization.cl')

        with open(fpath, 'r') as f:
            src = f.read()
            self.prg = cl.Program(self.ctx, src).build(options=compile_options)

        size = num_of_mom
        h_pol = np.zeros((size, 4), np.float32)
        h_rho = np.zeros(size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf = np.zeros((size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol)
        self.d_rho = cl_array.to_device(self.queue, h_rho)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf)

        self.prg.polarization_on_sf(self.queue, (block_size*num_of_mom,),
            (block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
            self.d_sf, self.d_omega, self.d_omega, d_momentum.data,
            np.int32(self.size_sf)).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf


if __name__ == '__main__':
    fpath = '/lustre/nyx/hyihp/lpang/auau200_results/cent20_30/etas0p08/event0/'
    sf = np.loadtxt(fpath+'/hypersf.dat', dtype=np.float32)
    omega = np.loadtxt(fpath+'/omegamu_sf.dat', dtype=np.float32).flatten()
    pol = Polarization(sf, omega)

    Y = np.linspace(-5, 5, 11, endpoint=True)
    px = np.linspace(-3, 3, 61, endpoint=True)
    py = np.linspace(-3, 3, 61, endpoint=True)


    mom_list = np.zeros((11*61*61, 4), dtype=np.float32)

    for i, Yi in enumerate(Y):
        for j, pxi in enumerate(px):
            for k, pyi in enumerate(py):
                mass = 1.0
                mt = math.sqrt(mass*mass + pxi*pxi + pyi*pyi)
                idx = i*len(px)*len(py) + j*len(py) + k
                mom_list[idx, 0] = mt
                mom_list[idx, 1] = Yi
                mom_list[idx, 2] = pxi
                mom_list[idx, 3] = pyi

    polar, density, pol_lrf = pol.get(mom_list)
    np.savetxt('polar.dat', polar)
    np.savetxt('density.dat', density)
    np.savetxt('polar_lrf.dat', pol_lrf)
