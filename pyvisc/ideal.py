#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time

from config import cfg

class CLIdeal(object):
    '''The pyopencl version for 3+1D ideal hydro dynamic simulation'''
    def __init__(self):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        #self.ctx = cl.create_some_context()
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        devices = [devices[0]]
        self.ctx = cl.Context(devices=devices, properties=[
            (cl.context_properties.PLATFORM, platform)])

        self.queue = cl.CommandQueue( self.ctx )

        self.size= np.int32(cfg.NX*cfg.NY*cfg.NZ)
        self.tau = cfg.real(cfg.TAU0)

    def __readIniCondition( self, fIni1 ):
        '''load initial condition (Ed, Umu ) from dat file '''
        try :
            dat1 = np.loadtxt(fIni1).astype(cfg.real)
        except IOError, e:
            print e
        self.h_ev1 = np.empty(self.size, cfg.real4)
        for i in range(self.size):
            self.h_ev1[ i ] = (dat1[i], 0.0, 0.0, 0.0) 

        #define buffer on device side,
        mf = cl.mem_flags
        self.d_ev1 = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, 
                hostbuf=self.h_ev1)
        self.d_ev2 = cl.Buffer( self.ctx, mf.READ_WRITE, self.h_ev1.nbytes )
        self.d_Src = cl.Buffer( self.ctx, mf.READ_WRITE, self.h_ev1.nbytes )

        self.submax = np.empty(64, cfg.real)
        self.d_submax = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.submax.nbytes)
        
    def __loadAndBuildCLPrg(self):
        optlist = [ 'DT', 'DX', 'DY', 'DZ', 'ETAOS', 'LAM1' ]
        self.gpu_defines = [ '-D %s=((real)%s)'%(key, value) for (key,value)
                in cfg.__dict__.items() if key in optlist ]
        self.gpu_defines.append('-D {key}={value}'.format(key='NX', value=cfg.NX))
        self.gpu_defines.append('-D {key}={value}'.format(key='NY', value=cfg.NY))
        self.gpu_defines.append('-D {key}={value}'.format(key='NZ', value=cfg.NZ))
        self.gpu_defines.append('-D {key}={value}'.format(key='SIZE', value=cfg.NX*cfg.NY*cfg.NZ))

        #local memory size along x,y,z direction with 4 boundary cells
        self.gpu_defines.append('-D {key}={value}'.format(key='BSZ', value=cfg.BSZ))
        self.gpu_defines.append('-D {key}={value}'.format(key='VSZ', value=cfg.BSZ-2))
        #determine float32 or double data type in *.cl file
        if cfg.use_float32:
            self.gpu_defines.append( '-D USE_SINGLE_PRECISION' )
        #choose EOS by ifdef in *.cl file
        if cfg.IEOS==0:
            self.gpu_defines.append( '-D EOSI' )
        elif cfg.IEOS==1:
            self.gpu_defines.append( '-D EOSLWB' )
        elif cfg.IEOS==2:
            self.gpu_defines.append( '-D EOSLPCE' )
        #set the include path for the header file
        cwd, cwf = os.path.split(__file__)
        self.gpu_defines.append('-I '+os.path.join(cwd, 'kernel/'))
        print self.gpu_defines
        #load and build *.cl programs with compile self.gpu_defines
        prg_src = open( os.path.join(cwd, 'kernel', 'kernel_ideal.cl'), 'r').read()
        self.kernel_ideal = cl.Program( self.ctx, prg_src ).build(options=self.gpu_defines)

        src_maxEd = open(os.path.join(cwd, 'kernel', 'kernel_reduction.cl'), 'r').read()
        self.kernel_reduction = cl.Program(self.ctx, src_maxEd).build(options=self.gpu_defines)

    def initHydro(self):
        '''Calc initial T^{tau mu} from initial Ed and Umu '''
        fname = cfg.fPathIni
        self.__loadAndBuildCLPrg()
        self.__readIniCondition( fname )

        #self.kernel_ideal.initIdeal( self.queue, (cfg.NX*cfg.NY*cfg.NZ, ), None, \
        #                    self.d_Tm0, self.d_Tm1, self.d_Um1, self.d_Src, \
        #                    self.d_Ed1, self.tau, self.size ).wait()
       
        # submax and d_submax is used to calc the maximum energy density


    def __stepUpdate(self, step):
        ''' Do step update in kernel with KT algorithm 
            Args:
                gpu_ev_old: self.d_ev1 for the 1st step,
			    self.d_ev2 for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
                      if ( step=1 and along_axis=1 ):
			 Initialize d_Src={0.0f}
                      else:
                         d_Src += src_from_kt1d
        '''
        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        mf = cl.mem_flags
        if step == 1: gpu_ev_old = self.d_ev1
        elif step == 2: gpu_ev_old = self.d_ev2
        # upadte d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src
        along_x, along_y, along_z = 1, 2, 3       # split along x, y, z
        self.kernel_ideal.kt_src_alongx(self.queue, (NX,NY,NZ), (NX, 1, 1), self.d_Src,
		       gpu_ev_old, self.tau, along_x, np.int32(step))

        self.kernel_ideal.kt_src_alongy(self.queue, (NX,NY,NZ), (1, NY, 1), self.d_Src,
                       gpu_ev_old, self.tau, np.int32(step))

        self.kernel_ideal.kt_src_alongz(self.queue, (NX,NY,NZ), (1, 1, NZ), self.d_Src,
                       gpu_ev_old, self.tau, np.int32(step))

        # if step=1, T0m' = T0m + d_Src*dt, update d_ev2
        # if step=2, T0m = T0m + 0.5*dt*d_Src, update d_ev1
        # Notice that d_Src=f(t,x) at step1 and 
        # d_Src=(f(t,x)+f(t+dt, x(t+dt))) at step2
        # input: gpu_ev_old to get T0m, d_Src, tau, size
        # output: T0m'->ed,v for 1st step and T0m->ed,v for 2nd step
        if step == 1:
	    self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ), None, self.d_ev2,
		 self.d_ev1, self.d_Src, self.tau, self.size, step)
        elif step == 2:
	    self.kernel_ideal.update_ev(self.queue, (NX*NY*NZ), None, self.d_ev1,
		 self.d_ev1, self.d_Src, self.tau, self.size, step)

    def __edMax(self):
        '''Calc the maximum energy density on GPU and output the value '''
        self.kernel_reduction.reduction_stage1(self.queue, (256*64,), (256,), 
                self.d_ev1, self.d_submax, self.size ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()


    def __output(self, nstep):
        if nstep%cfg.ntskip == 0:
            cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev1).wait()
            fout = '{pathout}/Ed{nstep}.dat'.format(
                    pathout=cfg.fPathOut, nstep=nstep)

            np.savetxt(fout, self.h_ev1.reshape(cfg.NX, cfg.NY, cfg.NZ)\
                    [::cfg.nxskip,::cfg.nyskip,::cfg.nzskip].flatten(), header='Ed')

            print nstep, ' finished'


    def evolve(self, ntskip=10):
        '''The main loop of hydrodynamic evolution '''
        for n in xrange(1000):
            self.__output(n)
            self.__stepUpdate(step=1)
            self.tau = cfg.real(cfg.TAU0 + (n+1)*cfg.DT)
            self.__stepUpdate(step=2)
            print 'EdMax= ',self.__edMax()
 


def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    ideal = CLIdeal()
    ideal.initHydro()
    ideal.evolve()
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format( dtime = t1-t0 )

if __name__ == '__main__':
    main()
