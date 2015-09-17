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

from config import cfg

def get_device_info(devices):
    print('image2d_max_width=', devices[0].image2d_max_width)
    print('local_mem_size=',    devices[0].local_mem_size)
    print('max_work_item_dimensions=', devices[0].max_work_item_dimensions)
    print('max_work_group_size=', devices[0].max_work_group_size)
    print('max_work_item_sizes=', devices[0].max_work_item_sizes)


class CLIdeal(object):
    '''The pyopencl version for 3+1D ideal hydro dynamic simulation'''
    def __init__(self):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        #self.ctx = cl.create_some_context()
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        devices = [devices[0]]

        get_device_info(devices)

        self.ctx = cl.Context(devices=devices, properties=[
            (cl.context_properties.PLATFORM, platform)])

        self.queue = cl.CommandQueue(self.ctx)

        self.size= cfg.NX*cfg.NY*cfg.NZ
        self.tau = cfg.real(cfg.TAU0)
        self.__loadAndBuildCLPrg()

        #define buffer on device side, d_ev1 stores ed, vx, vy, vz
        mf = cl.mem_flags
        self.h_ev1 = np.zeros((self.size, 4), cfg.real)

        # d_ev[0/1/2]: old/current/new value at time step n-1/n/n+1
        self.d_ev = [cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes),
                     cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)]

        self.d_Src = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)

        self.submax = np.empty(64, cfg.real)
        self.d_submax = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.submax.nbytes)

        self.history = []
 
    def read_ini(self, fIni1):
        '''load initial condition (Ed, vx, vy, vz) from dat file
           initial condition stored in 4 columns
           num_of_rows = NX*NY*NZ'''
        print('start to load ini data')
        dat1 = np.loadtxt(fIni1).astype(cfg.real)
        self.h_ev1 = dat1
        cl.enqueue_copy(self.queue, self.d_ev[1], self.h_ev1).wait()
        print('end of loading ini data')

       
    def __loadAndBuildCLPrg(self):
        optlist = [ 'DT', 'DX', 'DY', 'DZ', 'ETAOS', 'LAM1' ]
        self.gpu_defines = [ '-D %s=%sf'%(key, value) for (key,value)
                in list(cfg.__dict__.items()) if key in optlist ]
        self.gpu_defines.append('-D {key}={value}'.format(key='NX', value=cfg.NX))
        self.gpu_defines.append('-D {key}={value}'.format(key='NY', value=cfg.NY))
        self.gpu_defines.append('-D {key}={value}'.format(key='NZ', value=cfg.NZ))
        self.gpu_defines.append('-D {key}={value}'.format(key='SIZE', value=cfg.NX*cfg.NY*cfg.NZ))

        #local memory size along x,y,z direction with 4 boundary cells
        self.gpu_defines.append('-D {key}={value}'.format(key='BSZ', value=cfg.BSZ))
        #determine float32 or double data type in *.cl file
        if cfg.use_float32:
            self.gpu_defines.append( '-D USE_SINGLE_PRECISION' )
        #choose EOS by ifdef in *.cl file
        if cfg.IEOS==0:
            self.gpu_defines.append( '-D EOSI' )
        elif cfg.IEOS==1:
            self.gpu_defines.append( '-D EOSLCE' )
        elif cfg.IEOS==2:
            self.gpu_defines.append( '-D EOSLPCE' )
        #set the include path for the header file
        cwd, cwf = os.path.split(__file__)
        self.gpu_defines.append('-I '+os.path.join(cwd, 'kernel/'))
        print(self.gpu_defines)
        #load and build *.cl programs with compile self.gpu_defines
        with open(os.path.join(cwd, 'kernel', 'kernel_ideal.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_ideal = cl.Program(self.ctx, prg_src).build(
                                            options=self.gpu_defines)

        with open(os.path.join(cwd, 'kernel', 'kernel_reduction.cl'), 'r') as f:
            src_maxEd = f.read()
            self.kernel_reduction = cl.Program(self.ctx, src_maxEd
                                     ).build(options=self.gpu_defines)

    @classmethod
    def roundUp(cls, value, multiple):
        '''This function rounds one integer up to the nearest multiple of another integer,
        to get the global work size (which are multiples of local work size) from NX, NY, NZ.
        '''
        remainder = value % multiple
        if remainder != 0:
            value += multiple - remainder
        return value

    def __stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm 
            Args:
                gpu_ev_old: self.d_ev[1] for the 1st step,
                            self.d_ev[2] for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
        '''
        #NX = self.roundUp(cfg.NX, cfg.BSZ)
        #NY = self.roundUp(cfg.NY, cfg.BSZ)
        #NZ = self.roundUp(cfg.NZ, cfg.BSZ)
        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        mf = cl.mem_flags
        # upadte d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src
        #self.kernel_ideal.kt_src_alongx.set_scalar_arg_dtypes(np.float32, np.int32)
        self.kernel_ideal.kt_src_alongx(self.queue, (NX,NY,NZ), (NX, 1, 1),
                        self.d_Src, self.d_ev[step], self.tau, np.int32(step))

        self.kernel_ideal.kt_src_alongy(self.queue, (NX,NY,NZ), (1, NY, 1),
                        self.d_Src, self.d_ev[step], self.tau, np.int32(step))

        self.kernel_ideal.kt_src_alongz(self.queue, (NX,NY,NZ), (1, 1, NZ),
                        self.d_Src, self.d_ev[step], self.tau, np.int32(step))

        # if step=1, T0m' = T0m + d_Src*dt, update d_ev[2]
        # if step=2, T0m = T0m + 0.5*dt*d_Src, update d_ev[1]
        # Notice that d_Src=f(t,x) at step1 and 
        # d_Src=(f(t,x)+f(t+dt, x(t+dt))) at step2
        # output: d_ev[need_update] where need_update=2 for step 1 and 1 for step 2
        need_update = 3 - step
        self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ,), None,
                              self.d_ev[need_update], self.d_ev[1], self.d_Src,
                              self.tau, np.int32(step))

    def __edMax(self):
        '''Calc the maximum energy density on GPU and output the value '''
        self.kernel_reduction.reduction_stage1(self.queue, (256*64,), (256,), 
                self.d_ev[1], self.d_submax, np.int32(self.size) ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()


    def __output(self, nstep):
        if nstep%cfg.ntskip == 0:
            cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev[1]).wait()
            fout = '{pathout}/Ed{nstep}.dat'.format(
                    pathout=cfg.fPathOut, nstep=nstep)
            edxy = self.h_ev1[:,0].reshape(cfg.NX, cfg.NY, cfg.NZ)[:,:,cfg.NZ/2]
            np.savetxt(fout, self.h_ev1[:,0].reshape(cfg.NX, cfg.NY, cfg.NZ)\
                    [::cfg.nxskip,::cfg.nyskip,::cfg.nzskip].flatten(), header='Ed')



    def evolve(self, max_loops=1000, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        for n in range(max_loops):
            self.edmax = self.__edMax()
            self.history.append([self.tau, self.edmax])
            print('tau=', self.tau, ' EdMax= ',self.edmax)
            #self.__output(n)

            self.__stepUpdate(step=1)
            # update tau=tau+dtau for the 2nd step in RungeKutta
            self.tau = cfg.real(cfg.TAU0 + (n+1)*cfg.DT)
            self.__stepUpdate(step=2)
 


def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    #os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
    print('start ...')
    t0 = time()
    ideal = CLIdeal()
    fname = cfg.fPathIni
    ideal.read_ini(fname)
    ideal.evolve()
    t1 = time()
    print('finished. Total time: {dtime}'.format( dtime = t1-t0 ))

if __name__ == '__main__':
    main()
