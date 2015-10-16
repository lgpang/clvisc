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
        self.__loadAndBuildCLPrg()

        self.size =self.ideal.size
        self.h_pi0  = np.zeros(10*self.size, self.cfg.real)

        mf = cl.mem_flags
        self.d_pi = [cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes) ]
        self.d_IS_src = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        # d_udx, d_udy, d_udz, d_udt are velocity gradients for viscous hydro
        # datatypes are real4
        self.d_udt = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        self.d_udx = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        self.d_udy = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        self.d_udz = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        # traceless and transverse check
        self.d_checkpi = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)

        cl.enqueue_copy(self.queue, self.d_pi[1], self.h_pi0).wait()


    def __loadAndBuildCLPrg(self):
        self.cwd, cwf = os.path.split(__file__)
        #load and build *.cl programs with compile options
        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc = cl.Program(self.ctx, src).build(options=self.compile_options)
        #with open(os.path.join(self.cwd, 'kernel', 'kernel_src2.cl'), 'r') as f:
        #    src = f.read()
        #    self.kernel_src2 = cl.Program(self.ctx, src).build(options=self.compile_options)
        pass
            
    def stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm 
        This function is for one time step'''
        self.ideal.stepUpdate(step)
        eos_table = self.ideal.eos_table

        print "ideal update finished"
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_visc.visc_src_christoffel(self.queue, (NX*NY*NZ,), None,
                self.d_IS_src, self.d_pi[1], self.ideal.d_ev[1], eos_table,
                self.ideal.tau, np.int32(step));

        self.kernel_visc.visc_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.d_IS_src, self.d_udx, self.d_pi[1], self.ideal.d_ev[1],
                eos_table, self.ideal.tau).wait()

        print "udx along x"

        self.kernel_visc.visc_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.d_IS_src, self.d_udy, self.d_pi[1], self.ideal.d_ev[1],
                eos_table, self.ideal.tau).wait()

        print "udy along y"
        self.kernel_visc.visc_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.d_IS_src, self.d_udz, self.d_pi[1], self.ideal.d_ev[1],
                eos_table, self.ideal.tau).wait()

        print "udz along z"
        self.kernel_visc.update_pimn(self.queue, (NX*NY*NZ,), None,
                self.d_checkpi, self.d_pi[3-step], self.d_pi[1],
                self.ideal.d_ev[0], self.ideal.d_ev[3-step],
                self.d_udx, self.d_udy, self.d_udz, self.ideal.d_Src,
                eos_table, self.ideal.tau, np.int32(step)
                ).wait()

        print "sigma"


    def __output(self, nt):
        pass

    def plot_sigma_traceless(self):
        cl.enqueue_copy(self.queue, self.ideal.h_ev1, self.d_checkpi).wait()
        edxy = self.ideal.h_ev1[:,0].reshape(self.cfg.NX, self.cfg.NY, self.cfg.NZ)[:,:,55]
        import matplotlib.pyplot as plt
        plt.imshow(edxy.T)
        plt.colorbar()
        plt.show()

    def evolve(self, max_loops=1000, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        for loop in xrange(max_loops):
            cl.enqueue_copy(self.queue, self.ideal.d_ev[0],
                            self.ideal.d_ev[1]).wait()
                            
            print(self.ideal.max_energy_density())
            self.stepUpdate(step=1)
            self.ideal.tau = self.cfg.real(self.cfg.TAU0 +
                    (loop+1)*self.cfg.DT)
            self.stepUpdate(step=2)
            if loop % ntskip == 0:
                self.plot_sigma_traceless()




if __name__ == '__main__':
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    from config import cfg
    import pandas as pd
    visc = CLVisc(cfg)
    dat = np.loadtxt(cfg.fPathIni)
    #dat = pd.read_csv(cfg.fPathIni, sep=' ', skiprows=1,
    #        header=None, dtype=cfg.real).values
    visc.ideal.load_ini(dat)
    visc.evolve(max_loops=40)
    #visc.ideal.evolve(max_loops=200)
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format(dtime = t1-t0)

