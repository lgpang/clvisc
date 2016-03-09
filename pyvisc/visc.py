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

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from ideal import CLIdeal


class CLVisc(object):
    '''The pyopencl version for 3+1D visc hydro dynamic simulation'''
    def __init__(self, configs, gpu_id=0):
        self.ideal = CLIdeal(configs, gpu_id)
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

        nbytes_edv = self.ideal.h_ev1.nbytes
        self.d_udx = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)
        self.d_udy = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)
        self.d_udz = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)

        # d_omega vorticity vector omega^{mu}= epsilon^{mu nu a b} u_nu d_a u_b

        self.d_omega = [cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv),
                        cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)]

        # get the vorticity on the freeze out hypersurface
        self.d_omega_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real4)

        h_num_of_sf = np.zeros(1, np.int32)
        self.d_num_of_vorticity = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_num_of_sf);

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

        # initialize pimn, umu[2] such that its value can be changed before
        # self.evolve() is called for bjorken_test and gubser_test
        self.IS_initialize()
        self.ideal_predict_for_first_step()

        self.d_pizz_o_ep = cl.Buffer(self.ctx, mf.READ_WRITE, self.cfg.NZ*self.cfg.sz_real)



    def __loadAndBuildCLPrg(self):
        self.cwd, cwf = os.path.split(__file__)
        #load and build *.cl programs with compile options
        if self.cfg.gubser_visc_test:
            self.compile_options.append('-D GUBSER_VISC_TEST')

        with open(os.path.join(self.cwd, 'kernel', 'kernel_IS.cl'), 'r') as f:
            src = f.read()
            self.kernel_IS = cl.Program(self.ctx, src).build(options=self.compile_options)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc = cl.Program(self.ctx, src).build(options=self.compile_options)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_checkpi.cl'), 'r') as f:
            src = f.read()
            self.kernel_checkpi = cl.Program(self.ctx, src).build(options=self.compile_options)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_vorticity.cl'), 'r') as f:
            src = f.read()
            self.kernel_vorticity = cl.Program(self.ctx, src).build(options=self.compile_options)
            
    def IS_initialize(self):
        '''initialize pi^{mu nu} tensor'''
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_IS.visc_initialize(self.queue, (NX*NY*NZ,), None,
                self.d_pi[1], self.d_goodcell, self.d_udiff, self.ideal.d_ev[1],
                self.ideal.tau, self.eos_table).wait()


    def create_ini_from_partons(self, fname, SIGR=0.6, SIGZ=0.6, KFACTOR=1.0):
        '''generate initial condition from a list of partons in fname,
        SIGR: the gaussian smearing width in transverse direction
        SIGZ: the gaussian smearing width along longitudinal direction
        KFACTOR: scale factor to fit dNch/deta in most central collisions'''
        from smearing import Smearing
        Smearing(self.cfg, self.ctx, self.queue, self.compile_options,
            self.ideal.d_ev[1], fname, self.eos_table, SIGR, SIGZ, KFACTOR)

    def smear_from_p4x4(self, p4x4, SIGR=0.6, SIGZ=0.6, KFACTOR=1.0, force_bjorken=False):
        '''generate initial condition from a list of partons given by p4x4,
        SIGR: the gaussian smearing width in transverse direction
        SIGZ: the gaussian smearing width along longitudinal direction
        KFACTOR: scale factor to fit dNch/deta in most central collisions
        force_bjorken: True to switch off longitudinal fluctuation (use mid-rapidity only)'''
        from smearing import SmearingP4X4
        SmearingP4X4(self.cfg, self.ctx, self.queue, self.compile_options,
            self.ideal.d_ev[1], p4x4, self.eos_table, SIGR, SIGZ, KFACTOR, force_bjorken)



    def check_pizz(self):
        '''initialize pi^{mu nu} tensor'''
        NZ = self.cfg.NZ

        self.kernel_checkpi.pizz_o_ep(self.queue, (NZ,), None,
                self.d_pizz_o_ep, self.d_pi[1], self.ideal.d_ev[1],
                self.eos_table).wait()

        pizz_o_ep = np.zeros(NZ, dtype=np.float32)
        cl.enqueue_copy(self.queue, pizz_o_ep, self.d_pizz_o_ep).wait()
        print('<pizz/(e+P)> =', pizz_o_ep[NZ//2])



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


    #@profile
    def IS_stepUpdate(self, step):
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        self.kernel_IS.visc_src_christoffel(self.queue, (NX*NY*NZ,), None,
                self.d_IS_src, self.d_pi[step], self.ideal.d_ev[step],
                self.ideal.tau, np.int32(step)).wait()

        self.kernel_IS.visc_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.d_IS_src, self.d_udx, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        self.kernel_IS.visc_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.d_IS_src, self.d_udy, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        self.kernel_IS.visc_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.d_IS_src, self.d_udz, self.d_pi[step], self.ideal.d_ev[step],
                self.eos_table, self.ideal.tau).wait()

        # for step==1, d_ev[2] is useless, since u_new = u_old + d_udiff
        # for step==2, d_ev[2] is used to calc u_new
        self.kernel_IS.update_pimn(self.queue, (NX*NY*NZ,), None,
                #needs_update      not_important    start_point   src_for_RK
                self.d_pi[3-step], self.d_goodcell, self.d_pi[1], self.d_pi[step],
                self.ideal.d_ev[1], self.ideal.d_ev[2], self.d_udiff,
                self.d_udx, self.d_udy, self.d_udz, self.d_IS_src,
                self.eos_table, self.ideal.tau, np.int32(step)
                ).wait()

    def get_vorticity(self, loop, step, save_data=False):
        '''calc vorticity vector omega^{mu} and store them '''
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
        self.kernel_vorticity.omegamu(self.queue, (NX*NY*NZ,), None,
                self.ideal.d_ev[1], self.ideal.d_ev[2], self.d_udiff,
                self.d_udx, self.d_udy, self.d_udz, self.d_omega[1],
                self.eos_table, self.ideal.tau, np.int32(step)
                ).wait()

        if save_data:
            h_omega = np.empty((NX*NY*NZ, 4), self.cfg.real)
            cl.enqueue_copy(self.queue, h_omega, self.d_omega[1]).wait()
            path_out = os.path.abspath(self.cfg.fPathOut)
            np.savetxt(path_out + '/omegamu_%d.dat'%loop, h_omega,
            header='omega^{tau} omega^{x} omega^{y} omega^{eta} for NX*NY*NZ cells')


    def update_udiff(self, d_ev0, d_ev1):
        '''get d_udiff = u_{visc}^{n} - u_{visc}^{n-1}, it is possible to 
        set d_udiff in analytical solution for viscous gubser test'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.kernel_IS.get_udiff(self.queue, (NX*NY*NZ,), None,
            self.d_udiff, d_ev0, d_ev1).wait()
                
    def get_hypersf(self, n, ntskip, is_finished):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX//nxskip, NY//nyskip, NZ//nzskip}
        Params:
            :param n: the time step number
            :param ntskip: time step interval for hypersf calc
            :param is_finished: if True, the last time interval for hypersf
                   calculation will be different'''
        if n == 0:
            cl.enqueue_copy(self.queue, self.ideal.d_ev_old,
                            self.ideal.d_ev[1]).wait()
            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()

            # initialize the vorticity vector omega_mu at tau=0 with 0s
            zeros = np.zeros((self.size, 4), self.cfg.real)
            cl.enqueue_copy(self.queue, self.d_omega[0], zeros).wait()

            self.tau_old = self.cfg.TAU0
        elif (n % ntskip == 0) or is_finished:
            nx = (self.cfg.NX-1)//self.cfg.nxskip + 1
            ny = (self.cfg.NY-1)//self.cfg.nyskip + 1
            nz = (self.cfg.NZ-1)//self.cfg.nzskip + 1
            tau_new = self.ideal.tau
            self.kernel_hypersf.visc_hypersf(self.queue, (nx, ny, nz), None,
                    self.ideal.d_hypersf, self.d_pi_sf, self.ideal.d_num_of_sf,
                    self.ideal.d_ev_old, self.ideal.d_ev[1],
                    self.d_pi_old, self.d_pi[1],
                    self.d_omega_sf, self.d_omega[0], self.d_omega[1],
                    self.cfg.real(self.tau_old), self.cfg.real(tau_new)).wait()

            # update with current tau and d_ev[1], d_pi[1] and d_omega[1]
            cl.enqueue_copy(self.queue, self.ideal.d_ev_old,
                            self.ideal.d_ev[1]).wait()
            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.d_omega[0], self.d_omega[1]).wait()
            self.tau_old = tau_new


    def save_pimn_sf(self, set_to_zero=False):
        '''save pimn information on freeze out hyper surface
        Params:
            :param set_to_zero: True to set pimn on surface to 0.0,
            in case eta/s=0, ideal evolution is switch on'''
        num_of_sf = self.ideal.num_of_sf
        ed = self.ideal.efrz
        pr = self.ideal.eos.f_P(ed)
        T = self.cfg.TFRZ
        const_for_deltaf = 1.0/(2.0*T**2*(ed + pr))

        pi_onsf = np.zeros(10*num_of_sf, dtype=self.cfg.real)
        if not set_to_zero:
            cl.enqueue_copy(self.queue, pi_onsf, self.d_pi_sf).wait()
        out_path = os.path.join(self.cfg.fPathOut, 'pimnsf.dat')
        print("pimn on frzsf is saved to ", out_path)

        comment_line = 'one_o_2TsqrEplusP=%.6e '%const_for_deltaf + \
               'pi00 01 02 03 11 12 13 22 23 33'
        np.savetxt(out_path, pi_onsf.reshape(num_of_sf, 10), fmt='%.6e',
                   header = comment_line)



    def save(self, save_hypersf = True, save_bulk = False, 
             save_pi = False, save_vorticity=False):
        self.ideal.save(save_hypersf, save_bulk)

        if save_pi:
            self.pimn_info.save()

        if save_hypersf:
            self.save_pimn_sf()

        if save_vorticity:
            # save vorticity on hypersf to data file
            omega_mu = np.empty(num_of_sf, dtype=self.cfg.real4)
            cl.enqueue_copy(self.queue, omega_mu, self.d_omega_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_sf.dat')
            print("vorticity omega_{mu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_mu, fmt='%.6e', header = 'omega_mu ')


    def update_time(self, loop):
        self.ideal.update_time(loop)


    def ideal_predict_for_first_step(self):
        # ideal prediction to get umu for the first time step
        self.ideal.stepUpdate(step=1)

    #@profile
    def evolve(self, max_loops=1000, save_hypersf=True, save_bulk=False,
               plot_bulk=True, save_pi=True, force_run_to_maxloop = False,
               save_vorticity=False):
        '''The main loop of hydrodynamic evolution
        default parameters: save_hypersf, don't save bulk info
        store bulk info by switch on plot_bulk'''
        # if etaos<1.0E-6, use ideal hydrodynamics which is much faster
        if self.cfg.ETAOS < 1.0E-6 and not save_vorticity:
            self.ideal.evolve(max_loops, save_hypersf, save_bulk,
                    plot_bulk, force_run_to_maxloop)

            self.save_pimn_sf(set_to_zero=True)
            return

        if save_pi:
            from pimninfo import PimnInfo
            self.pimn_info = PimnInfo(self.cfg, self.ctx, self.queue,
                                 self.compile_options)

        for loop in xrange(max_loops):
            #t0 = time()
            self.ideal.edmax = self.ideal.max_energy_density()
            self.ideal.history.append([self.ideal.tau, self.ideal.edmax])
            print('tau=', self.ideal.tau, ' EdMax= ',self.ideal.edmax)

            is_finished = self.ideal.edmax < self.ideal.efrz

            if save_hypersf:
                self.get_hypersf(loop, self.cfg.ntskip, is_finished)

            if is_finished and not force_run_to_maxloop:
                break

            if (plot_bulk or save_bulk) and loop % self.cfg.ntskip == 0:
                self.ideal.bulkinfo.get(self.ideal.tau,
                        self.ideal.d_ev[1], self.ideal.edmax)

            if save_pi and loop % self.cfg.ntskip == 0:
                self.pimn_info.get(self.ideal.tau, self.d_pi[1])

            # store d_pi[0] for self.visc_stepUpdate()
            cl.enqueue_copy(self.queue, self.d_pi[0],
                            self.d_pi[1]).wait()

            # copy the d_ev[1] to d_ev[0] for umu_new prediction
            cl.enqueue_copy(self.queue, self.ideal.d_ev[0],
                            self.ideal.d_ev[1]).wait()

            # update pi[2] with d_ev[0] and u_new=u0+d_udiff
            # where d_udiff is prediction from previous step
            self.IS_stepUpdate(step=1)
            self.visc_stepUpdate(step=1)
            self.update_time(loop)
            # update pi[1] with d_ev[0] and d_ev[2]_visc*

            # remove the following 2 lines to speed up without vorticity calculation
            if loop % self.cfg.ntskip == 0 and save_vorticity:
                self.get_vorticity(loop, step=2, save_data=False)

            self.IS_stepUpdate(step=2)
            self.visc_stepUpdate(step=2)
            self.update_udiff(self.ideal.d_ev[0], self.ideal.d_ev[1])

        self.save(save_hypersf=save_hypersf, save_bulk=save_bulk, 
                  save_pi = save_pi, save_vorticity = save_vorticity)



def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    from config import cfg, write_config
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 61

    cfg.DT = 0.005
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.ImpactParameter = 10.0
    cfg.IEOS = 1
    cfg.ntskip = 60
    cfg.nxskip = 2
    cfg.nyskip = 2
    cfg.nzskip = 1

    cfg.ETAOS = 0.08
    write_config(cfg)

    visc = CLVisc(cfg, gpu_id=0)
    from glauber import Glauber
    Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
            visc.ideal.d_ev[1])

    visc.evolve(max_loops=2000)
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format(dtime = t1-t0)


if __name__ == '__main__':
    main()
