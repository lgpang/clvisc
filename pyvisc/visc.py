#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time

from ideal import CLIdeal

class CLVisc(object):
    '''The pyopencl version for 3+1D visc hydro dynamic simulation'''
    def __init__(self, configs, gpu_id=0):
        self.ideal = CLIdeal(configs, gpu_id, viscous_on=True)
        self.cfg = configs
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue
        self.compile_options = self.ideal.gpu_defines

        self.cwd = self.ideal.cwd
        self.size =self.ideal.size
        self.h_pi0  = np.empty(10*self.size, self.cfg.real)

        mf = cl.mem_flags
        self.d_pi = [cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                    cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes) ]

    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile options

        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc = cl.Program(self.ctx, src).build(options=self.compile_options)

        #with open(os.path.join(self.cwd, 'kernel', 'kernel_src2.cl'), 'r') as f:
        #    src = f.read()
        #    self.kernel_src2 = cl.Program(self.ctx, src).build(options=self.compile_options)

    def __stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm 
        This function is for one time step'''
        pass

    def __output(self, nt):
        pass

    def evolve(self, max_loops=1000, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        pass



if __name__ == '__main__':
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    from config import cfg
    dat = np.loadtxt(cfg.fPathIni)
    visc = CLVisc(cfg)
    visc.ideal.load_ini(dat)
    visc.ideal.evolve(max_loops=200)
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format( dtime = t1-t0 )

