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
        self.h_goodcell = np.ones(self.size, self.cfg.real)
        self.d_goodcell = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_goodcell.nbytes)

        cl.enqueue_copy(self.queue, self.d_pi[1], self.h_pi0).wait()
        cl.enqueue_copy(self.queue, self.d_goodcell, self.h_goodcell).wait()

        # used for freeze out hypersurface calculation
        self.d_pi_old = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)

        # store the pi^{mu nu} on freeze out hyper surface
        self.d_pi_sf = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        self.kernel_hypersf = self.ideal.kernel_hypersf

        # initialize pimn such that its value can be changed before
        # self.evolve() is called for bjorken_test and gubser_test
        self.IS_initialize()

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
            
    def IS_initialize(self):
        '''initialize pi^{mu nu} tensor'''
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_IS.visc_initialize(self.queue, (NX*NY*NZ,), None,
                self.d_pi[1], self.d_goodcell, self.d_udiff, self.ideal.d_ev[1],
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
                self.d_pi[3-step], self.d_goodcell, self.d_pi[1], self.d_pi[step],
                self.ideal.d_ev[1], self.ideal.d_ev[2], self.d_udiff,
                self.d_udx, self.d_udy, self.d_udz, self.d_IS_src,
                self.eos_table, self.ideal.tau, np.int32(step)
                ).wait()


    def update_udiff(self):
        '''get d_udiff = u_{visc}^{n} - u_{ideal*}^{n} '''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.kernel_IS.get_udiff(self.queue, (NX*NY*NZ,), None,
            self.d_udiff, self.ideal.d_ev[0], self.ideal.d_ev[1]).wait()
                
    def get_hypersf(self, n, ntskip):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX//nxskip, NY//nyskip, NZ//nzskip} '''
        is_finished = self.ideal.edmax < self.ideal.efrz

        if n == 0:
            cl.enqueue_copy(self.queue, self.ideal.d_ev_old,
                            self.ideal.d_ev[1]).wait()
            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()
            self.tau_old = self.cfg.TAU0
        elif (n % ntskip == 0) or is_finished:
            nx = (self.cfg.NX-1)//self.cfg.nxskip + 1
            ny = (self.cfg.NY-1)//self.cfg.nyskip + 1
            nz = (self.cfg.NZ-1)//self.cfg.nzskip + 1
            tau_new = self.tau
            self.kernel_hypersf.visc_hypersf(self.queue, (nx, ny, nz), None,
                    self.ideal.d_hypersf, self.d_pi_sf, self.ideal.d_num_of_sf,
                    self.ideal.d_ev_old, self.ideal.d_ev[1],
                    self.d_pi_old, self.d_pi[1],
                    self.cfg.real(self.tau_old), self.cfg.real(tau_new)).wait()

            # update with current tau and d_ev[1]
            cl.enqueue_copy(self.queue, self.ideal.d_ev_old,
                            self.ideal.d_ev[1]).wait()
            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()
            self.tau_old = tau_new

        return is_finished

    def save(self, save_hypersf=True, save_bulk=True):
        self.ideal.save(save_hypersf, save_bulk)
        if save_hypersf:
            pi_onsf = np.empty(10*self.num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, pi_onsf, self.d_pi_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'pimnsf.dat')
            print("pimn on frzsf is saved to ", out_path)
            np.savetxt(out_path, pi_onsf.reshape(self.num_of_sf, 10),
                       header = 'pi00 01 02 03 11 12 13 22 23 33')

    def update_time(self, loop):
        self.ideal.update_time(loop)

    #@profile
    def evolve(self, max_loops=1000, save_hypersf=True, save_bulk=True,
               to_maxloop = False):
        '''The main loop of hydrodynamic evolution '''
        for loop in xrange(max_loops):
            self.ideal.edmax = self.ideal.max_energy_density()
            self.ideal.history.append([self.ideal.tau, self.ideal.edmax])
            print('tau=', self.ideal.tau, ' EdMax= ',self.ideal.edmax)

            is_finished = False

            if save_hypersf:
                is_finished = self.get_hypersf(loop, self.cfg.ntskip)

            if is_finished and not to_maxloop:
                break

            if save_bulk and loop % self.cfg.ntskip == 0:
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

        self.save(save_hypersf=save_hypersf, save_bulk=save_bulk)



def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    from config import cfg
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 61

    cfg.DT = 0.005
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.ImpactParameter = 10.0
    cfg.IEOS = 2
    cfg.ntskip = 100

    cfg.ETAOS = 0.16

    visc = CLVisc(cfg)
    from glauber import Glauber
    Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
            visc.ideal.d_ev[1])

    visc.evolve(max_loops=2000)
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format(dtime = t1-t0)

if __name__ == '__main__':
    main()
