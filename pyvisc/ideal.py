#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time
#import matplotlib.pyplot as plt


def get_device_info(devices):
    print('image2d_max_width=', devices[0].image2d_max_width)
    print('local_mem_size=',    devices[0].local_mem_size)
    print('max_work_item_dimensions=', devices[0].max_work_item_dimensions)
    print('max_work_group_size=', devices[0].max_work_group_size)
    print('max_work_item_sizes=', devices[0].max_work_item_sizes)


class CLIdeal(object):
    '''The pyopencl version for 3+1D ideal hydro dynamic simulation'''
    def __init__(self, configs, gpu_id=0, viscous_on=False):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        self.cfg = configs
        self.cwd, cwf = os.path.split(__file__)

        from backend_opencl import OpenCLBackend
        backend = OpenCLBackend(self.cfg, gpu_id)

        self.ctx = backend.ctx
        self.queue = backend.default_queue

        self.size= self.cfg.NX*self.cfg.NY*self.cfg.NZ
        self.tau = self.cfg.real(self.cfg.TAU0)

        # set viscous on to cal fluid velocity gradients
        self.viscous_on = viscous_on
        self.gpu_defines = self.__compile_options()
        self.__loadAndBuildCLPrg()

        #define buffer on device side, d_ev1 stores ed, vx, vy, vz
        mf = cl.mem_flags
        self.h_ev1 = np.zeros((self.size, 4), self.cfg.real)

        # d_ev[0/1/2]: old/current/new value at time step n-1/n/n+1
        self.d_ev = [cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)]

        self.d_Src = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)

        self.submax = np.empty(64, self.cfg.real)
        self.d_submax = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.submax.nbytes)
        # d_ev_old: for hypersf calculation; 
        self.d_ev_old = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)
        # d_hypersf: store the dSigma^{mu}, vx, vy, veta, tau, x, y, eta
        # on freeze out hyper surface
        self.d_hypersf = cl.Buffer(self.ctx, mf.READ_WRITE, size=1000000*self.cfg.sz_real8)
        h_num_of_sf = np.zeros(1, np.int32)
        self.d_num_of_sf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_num_of_sf);

        self.history = []
 
    def load_ini(self, dat):
        '''load initial condition stored in np array whose 4 columns
           are (Ed, vx, vy, vz) and  num_of_rows = NX*NY*NZ'''
        print('start to load ini data')
        self.h_ev1 = dat.astype(self.cfg.real)
        cl.enqueue_copy(self.queue, self.d_ev[1], self.h_ev1).wait()
        print('end of loading ini data')

    def __compile_options(self):
        optlist = [ 'DT', 'DX', 'DY', 'DZ', 'ETAOS', 'LAM1' ]
        gpu_defines = [ '-D %s=%sf'%(key, value) for (key,value)
                in list(self.cfg.__dict__.items()) if key in optlist ]
        gpu_defines.append('-D {key}={value}'.format(key='NX', value=self.cfg.NX))
        gpu_defines.append('-D {key}={value}'.format(key='NY', value=self.cfg.NY))
        gpu_defines.append('-D {key}={value}'.format(key='NZ', value=self.cfg.NZ))
        gpu_defines.append('-D {key}={value}'.format(key='SIZE',
                           value=self.cfg.NX*self.cfg.NY*self.cfg.NZ))

        #local memory size along x,y,z direction with 4 boundary cells
        gpu_defines.append('-D {key}={value}'.format(key='BSZ', value=self.cfg.BSZ))
        if self.viscous_on:
            gpu_defines.append( '-D VISCOUS_ON' )
        #determine float32 or double data type in *.cl file
        if self.cfg.use_float32:
            gpu_defines.append( '-D USE_SINGLE_PRECISION' )
        #choose EOS by ifdef in *.cl file
        if self.cfg.IEOS==0:
            gpu_defines.append( '-D EOSI' )
        elif self.cfg.IEOS==1:
            gpu_defines.append( '-D EOSLCE' )
        elif self.cfg.IEOS==2:
            gpu_defines.append( '-D EOSLPCE' )
        #set the include path for the header file
        gpu_defines.append('-I '+os.path.join(self.cwd, 'kernel/'))
        return gpu_defines
      
    def __loadAndBuildCLPrg(self):
        print(self.gpu_defines)
        #load and build *.cl programs with compile self.gpu_defines
        with open(os.path.join(self.cwd, 'kernel', 'kernel_ideal.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_ideal = cl.Program(self.ctx, prg_src).build(
                                             options=self.gpu_defines)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_reduction.cl'), 'r') as f:
            src_maxEd = f.read()
            self.kernel_reduction = cl.Program(self.ctx, src_maxEd).build(
                                                 options=self.gpu_defines)

        hypersf_defines = list(self.gpu_defines)
        hypersf_defines.append('-D {key}={value}'.format(key='nxskip', value=self.cfg.nxskip))
        hypersf_defines.append('-D {key}={value}'.format(key='nyskip', value=self.cfg.nyskip))
        hypersf_defines.append('-D {key}={value}'.format(key='nzskip', value=self.cfg.nzskip))
        hypersf_defines.append('-D {key}={value}f'.format(key='EFRZ', value=0.25))
        print(hypersf_defines)
        with open(os.path.join(self.cwd, 'kernel', 'kernel_hypersf.cl'), 'r') as f:
            src_hypersf = f.read()
            self.kernel_hypersf = cl.Program(self.ctx, src_hypersf).build(
                                                 options=hypersf_defines)



    @classmethod
    def roundUp(cls, value, multiple):
        '''This function rounds one integer up to the nearest multiple of another integer,
        to get the global work size (which are multiples of local work size) from NX, NY, NZ.
        '''
        remainder = value % multiple
        if remainder != 0:
            value += multiple - remainder
        return value

    def stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm 
            Args:
                gpu_ev_old: self.d_ev[1] for the 1st step,
                            self.d_ev[2] for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
        '''
        # upadte d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
        self.kernel_ideal.kt_src_christoffel(self.queue, (NX*NY*NZ, ), None,
                         self.d_Src, self.d_ev[step], self.tau, np.int32(step)
                         ).wait()

        self.kernel_ideal.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.d_Src, self.d_ev[step], self.tau).wait()

        self.kernel_ideal.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.d_Src, self.d_ev[step], self.tau).wait()

        self.kernel_ideal.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.d_Src, self.d_ev[step], self.tau).wait()

        # if step=1, T0m' = T0m + d_Src*dt, update d_ev[2]
        # if step=2, T0m = T0m + 0.5*dt*d_Src, update d_ev[1]
        # Notice that d_Src=f(t,x) at step1 and 
        # d_Src=(f(t,x)+f(t+dt, x(t+dt))) at step2
        # output: d_ev[] where need_update=2 for step 1 and 1 for step 2
        self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ, ), None,
                              self.d_ev[3-step], self.d_ev[1], self.d_Src,
                              self.tau, np.int32(step)).wait()

    def max_energy_density(self):
        '''Calc the maximum energy density on GPU and output the value '''
        self.kernel_reduction.reduction_stage1(self.queue, (256*64,), (256,), 
                self.d_ev[1], self.d_submax, np.int32(self.size) ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()


    def output(self, nstep):
        if nstep%self.cfg.ntskip == 0:
            cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev[1]).wait()
            fout = '{pathout}/Ed{nstep}.dat'.format(
                    pathout=self.cfg.fPathOut, nstep=nstep)
            edxy = self.h_ev1[:,1].reshape(self.cfg.NX, self.cfg.NY, self.cfg.NZ)[:,:,self.cfg.NZ//2]
            #np.savetxt(fout, self.h_ev1[:,0].reshape(self.cfg.NX, self.cfg.NY, self.cfg.NZ)
            #        [::self.cfg.nxskip,::self.cfg.nyskip,::self.cfg.nzskip].flatten(), header='Ed, vx, vy, veta')
            #plt.imshow(edxy)
            #plt.show()

    def get_hypersf(self, n, ntskip, tau_old):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX//nxskip, NY//nyskip, NZ//nzskip} '''
        tau_new = self.cfg.TAU0 + n*self.cfg.DT
        nx = self.cfg.NX//self.cfg.nxskip
        ny = self.cfg.NY//self.cfg.nyskip
        nz = self.cfg.NZ//self.cfg.nzskip
        is_finished = self.edmax < 0.25

        if ( (n % ntskip == 0 and n != 0) or is_finished):
            self.kernel_hypersf.get_hypersf(self.queue, (nx, ny, nz), None,
                    self.d_hypersf, self.d_num_of_sf, self.d_ev_old, self.d_ev[1],
                    self.cfg.real(tau_old), self.cfg.real(tau_new)).wait()

            cl.enqueue_copy(self.queue, self.d_ev_old, self.d_ev[1]).wait()
            self.num_of_sf = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, self.num_of_sf, self.d_num_of_sf).wait()
            print("num of sf=", self.num_of_sf)

        if ( is_finished ):
            hypersf = np.empty(self.num_of_sf, dtype=self.cfg.real8)
            cl.enqueue_copy(self.queue, hypersf, self.d_hypersf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'hypersf.dat')
            print("hypersf save to ", out_path)
            np.savetxt(out_path, hypersf, header = 'dS0, dS1, dS2, dS3, vx, vy, veta, etas')
            exit()


    def evolve(self, max_loops=1000, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        cl.enqueue_copy(self.queue, self.d_ev_old, self.d_ev[1]).wait()
        tau_old = self.cfg.TAU0
        for n in range(max_loops):
            self.edmax = self.max_energy_density()
            self.history.append([self.tau, self.edmax])
            print('tau=', self.tau, ' EdMax= ',self.edmax)
            self.get_hypersf(n, ntskip, tau_old)
            #self.output(n)

            self.stepUpdate(step=1)
            # update tau=tau+dtau for the 2nd step in RungeKutta
            self.tau = self.cfg.real(self.cfg.TAU0 + (n+1)*self.cfg.DT)
            self.stepUpdate(step=2)

           
 


def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    #os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
    from config import cfg
    #import pandas as pd
    print('start ...')
    t0 = time()
    ideal = CLIdeal(cfg)
    from glauber import Glauber
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
                  ideal.d_ev[1])
    #dat = np.loadtxt(cfg.fPathIni)
    #ideal.load_ini(dat)
    ideal.evolve()
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0 ))

if __name__ == '__main__':
    main()
