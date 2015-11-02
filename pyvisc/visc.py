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

        self.eos_table = self.ideal.eos_table

        self.size =self.ideal.size
        self.h_pi0  = np.zeros(10*self.size, self.cfg.real)

        mf = cl.mem_flags
        self.d_pi = [cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes) ]
        self.d_IS_src = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        # d_udx, d_udy, d_udz, d_udt are velocity gradients for viscous hydro
        # datatypes are real4
        self.d_udx = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        self.d_udy = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)
        self.d_udz = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)

        # velocity difference between u_visc and u_ideal* for correction
        self.d_udiff = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)

        # traceless and transverse check
        # self.d_checkpi = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.ideal.h_ev1.nbytes)

        cl.enqueue_copy(self.queue, self.d_pi[1], self.h_pi0).wait()


    def __loadAndBuildCLPrg(self):
        self.cwd, cwf = os.path.split(__file__)
        #load and build *.cl programs with compile options
        with open(os.path.join(self.cwd, 'kernel', 'kernel_IS.cl'), 'r') as f:
            src = f.read()
            self.kernel_IS = cl.Program(self.ctx, src).build(options=self.compile_options)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc = cl.Program(self.ctx, src).build(options=self.compile_options)
        pass
            
    def initialize(self):
        '''initialize pi^{mu nu} tensor'''
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_IS.visc_initialize(self.queue, (NX*NY*NZ,), None,
                self.d_pi[1], self.d_udiff, self.ideal.d_ev[1],
                self.ideal.tau, self.eos_table).wait()


    def visc_stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm for visc evolution
            Args:
                gpu_ev_old: self.d_ev[1] for the 1st step,
                            self.d_ev[2] for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
        '''
        # upadte d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
        self.kernel_visc.kt_src_christoffel(self.queue, (NX*NY*NZ, ), None,
                         self.ideal.d_Src, self.ideal.d_ev[step],
                         self.d_pi[step], self.eos_table,
                         self.ideal.tau, np.int32(step)
                         ).wait()

        self.kernel_visc.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.ideal.d_Src, self.ideal.d_ev[step],
                self.d_pi[step], self.eos_table,
                self.ideal.tau).wait()

        self.kernel_visc.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.ideal.d_Src, self.ideal.d_ev[step],
                self.d_pi[step], self.eos_table,
                self.ideal.tau).wait()

        self.kernel_visc.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.ideal.d_Src, self.ideal.d_ev[step],
                self.d_pi[step], self.eos_table,
                self.ideal.tau).wait()

        # if step=1, T0m' = T0m + d_Src*dt, update d_ev[2]
        # if step=2, T0m = T0m + 0.5*dt*d_Src, update d_ev[1]
        # Notice that d_Src=f(t,x) at step1 and 
        # d_Src=(f(t,x)+f(t+dt, x(t+dt))) at step2
        # output: d_ev[] where need_update=2 for step 1 and 1 for step 2
        self.kernel_visc.update_ev(self.queue, (NX*NY*NZ, ), None,
                              self.ideal.d_ev[3-step], self.ideal.d_ev[1],
                              self.d_pi[0], self.d_pi[3-step],
                              self.ideal.d_Src,
                              self.eos_table, self.ideal.tau, np.int32(step)).wait()


    def IS_stepUpdate(self, step):
        #print "ideal update finished"
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_IS.visc_src_christoffel(self.queue, (NX*NY*NZ,), None,
                self.d_IS_src, self.d_pi[step], self.ideal.d_ev[step],
                self.ideal.tau, np.int32(step)).wait()

        self.kernel_IS.visc_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.d_IS_src, self.d_udx, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        #print "udx along x"

        self.kernel_IS.visc_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.d_IS_src, self.d_udy, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        #print "udy along y"
        self.kernel_IS.visc_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.d_IS_src, self.d_udz, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        #print "udz along z"
        self.kernel_IS.update_pimn(self.queue, (NX*NY*NZ,), None,
                self.d_pi[3-step], self.d_pi[1], self.d_pi[step],
                self.ideal.d_ev[1], self.ideal.d_ev[2], self.d_udiff,
                self.d_udx, self.d_udy, self.d_udz, self.d_IS_src,
                self.eos_table, self.ideal.tau, np.int32(step)
                ).wait()


    def update_udiff(self):
        '''get d_udiff = u_{visc}^{n} - u_{ideal*}^{n} '''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.kernel_IS.get_udiff(self.queue, (NX*NY*NZ,), None,
            self.d_udiff, self.ideal.d_ev[0], self.ideal.d_ev[1]).wait()
                

    def __output(self, nt):
        pass

    def plot_sigma_traceless(self, i):
        cl.enqueue_copy(self.queue, self.ideal.h_ev1, self.ideal.d_ev[1]).wait()
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        edxy = self.ideal.h_ev1[:, 1].reshape(NX, NY, NZ)[:,:,NZ/2]
        #np.savetxt('debug/pi_traceless%d.dat'%i, edxy)
        import matplotlib.pyplot as plt
        plt.imshow(edxy.T)
        plt.colorbar()
        plt.show()

    def update_time(self, loop):
        self.ideal.update_time(loop)

    def IS_test(self, max_loops=1000, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        self.initialize()
        for loop in xrange(max_loops):
            cl.enqueue_copy(self.queue, self.ideal.d_ev[0],
                            self.ideal.d_ev[1]).wait()
            ''' Do step update in kernel with KT algorithm 
            This function is for one time step'''
            self.ideal.stepUpdate(step=1)
            #print(self.ideal.max_energy_density())
            self.IS_stepUpdate(step=1)
            self.update_time(loop)
            self.IS_stepUpdate(step=2)
            #if loop % ntskip == 0:
            #self.plot_sigma_traceless(loop)
            print('loop=', loop)

    def evolve(self, max_loops=1000, save_hypersf=True, save_bulk=True):
        '''The main loop of hydrodynamic evolution '''
        self.initialize()
        for loop in xrange(max_loops):
            self.ideal.edmax = self.ideal.max_energy_density()
            self.ideal.history.append([self.ideal.tau, self.ideal.edmax])
            print('tau=', self.ideal.tau, ' EdMax= ',self.ideal.edmax)
            is_finished = self.ideal.get_hypersf(loop, self.cfg.ntskip)
            if is_finished:
                break

            if loop % self.cfg.ntskip == 0:
                self.ideal.bulkinfo.get(self.ideal.tau,
                        self.ideal.d_ev[1], self.ideal.edmax)

            # store d_pi[0]
            cl.enqueue_copy(self.queue, self.d_pi[0],
                            self.d_pi[1]).wait()
            # ideal prediction; d_ev[2] is updated
            self.ideal.stepUpdate(step=1)

            # copy the ideal prediction to d_ev[0] for d_udiff calc
            cl.enqueue_copy(self.queue, self.ideal.d_ev[0],
                            self.ideal.d_ev[2]).wait()

            # update pi[2] with d_ev[0] and d_ev[2]_ideal
            # the difference is corrected with d_udiff
            self.IS_stepUpdate(step=1)
            self.visc_stepUpdate(step=1)
            self.update_time(loop)
            # update pi[1] with d_ev[0] and d_ev[2]_visc
            self.IS_stepUpdate(step=2)
            self.visc_stepUpdate(step=2)
            self.update_udiff()

            if loop % self.cfg.ntskip == 0:
                #self.plot_sigma_traceless(loop)
                pass

        self.ideal.save(save_hypersf=save_hypersf, save_bulk=save_bulk)



def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    from config import cfg
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 1

    cfg.DT = 0.02
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.ImpactParameter = 0.0
    cfg.IEOS = 2
    cfg.ntskip = 100

    cfg.ETAOS = 0.08

    visc = CLVisc(cfg)
    from glauber import Glauber
    Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
            visc.ideal.d_ev[1])

    #visc.IS_test(max_loops=80)
    visc.evolve(max_loops=2000)
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format(dtime = t1-t0)

if __name__ == '__main__':
    main()
