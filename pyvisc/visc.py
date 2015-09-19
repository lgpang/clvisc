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
from ideal import CLIdeal
from backend_opencl import OpenCLBackend

class CLVisc(object):
    '''The pyopencl version for 3+1D visc hydro dynamic simulation'''
    def __init__(self, config, backend):
        self.ideal = CLIdeal(configs=cfg, backend)
        self.ctx = backend.ctx
        self.queue = backend.default_queue
        self.h_pi0  = np.empty(10*self.size, cfg.real)

        self.d_pi0 = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        self.d_pi1 = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        self.d_pi2 = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)

    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile options
        src = open( 'kernel/kernel_visc.cl', 'r').read()
        self.kernel_visc = cl.Program(self.ctx, src).build(options=self.gpu_defines)

        src = open('kernel/kernel_src2.cl', 'r').read()
        self.kernel_src2 = cl.Program(self.ctx, src).build(options=self.gpu_defines)



    def initHydro( self ):
        '''Calc initial T^{tau mu} from initial Ed and Umu '''
        fname = cfg.fPathIni
        self.__loadAndBuildCLPrg()
        self.__readIniCondition( fname )
        super(CLVisc, self).initHydro()
        LSZ = cfg.BSZ - 4

        self.kernel_visc.initVisc( self.queue, (cfg.NX*cfg.NY*cfg.NZ,), None, \
                            self.d_pi0, self.d_pi1, self.d_Ed1, self.d_um0, \
                            self.d_um1, self.tau, self.size ).wait()



    def __stepUpdate( self, halfStep=np.int32(1) ):
        ''' Do step update in kernel with KT algorithm 
        This function is for one time step'''
        NX,NY,NZ = cfg.NX, cfg.NY, cfg.NZ
        BSZ = 7
        LSZ = BSZ - 4

        mf = cl.mem_flags

        global_size, local_size = (NX,NY,NZ), (LSZ, LSZ, LSZ)

        if halfStep==1 :
            #update source term 
            self.kernel_visc.stepUpdateVisc1( self.queue, global_size, local_size, \
                self.d_pi0, self.d_pi1, self.d_um0, self.d_um1, self.d_Ed1, self.d_pi2, \
                self.d_Src, self.d_Sigma, self.tau, halfStep, self.size ).wait()

            #update KT3D
            self.kernel_visc.stepUpdateVisc( self.queue, global_size, local_size, \
                self.d_pi1, self.d_pi1, self.d_um0, self.d_um1, self.d_Ed1, self.d_pi2, \
                self.d_Src, self.tau, halfStep, self.size ).wait()

            #update src2
            self.kernel_src2.updateSrcFromPimn(self.queue, global_size, local_size, \
                    self.d_pi2, self.d_um1, self.d_Src, self.tau, self.size)
        else:
            #update source term 
            self.kernel_visc.stepUpdateVisc1( self.queue, global_size, local_size, \
                self.d_pi1, self.d_pi2, self.d_um1, self.d_NewUmu, self.d_NewEd, self.d_pi1, \
                self.d_Src, self.d_Sigma, self.tau, halfStep, self.size ).wait()

            #update KT3D
            self.kernel_visc.stepUpdateVisc( self.queue, global_size, local_size, \
                self.d_pi2, self.d_pi2, self.d_um1, self.d_NewUmu, self.d_NewEd, self.d_pi1, \
                self.d_Src, self.tau, halfStep, self.size ).wait()

            #update src2
            self.kernel_src2.updateSrcFromPimn(self.queue, global_size, local_size, \
                    self.d_pi1,  self.d_NewUmu, self.d_Src, cfg.real(self.tau+cfg.DT), self.size)

        super(CLVisc, self).__stepUpdate(halfStep)


    def __updateGlobalMem( self, halfStep=np.int32(1)):
        ''' A->A*; A*->A**;   Anew = 0.5*(A+A**); update d_Tm01, Ed, Umu at 
        last step of RungeKuta method'''
        mf = cl.mem_flags
        d_Nreg = cl.Buffer(self.ctx, mf.READ_WRITE, cfg.size_real)
        self.kernel_visc.updateGlobalMemVisc( self.queue, (cfg.NX*cfg.NY*cfg.NZ,), None, \
                self.d_pi0, self.d_pi1, self.d_um0, self.d_um1, self.d_pi2, self.d_Ed1, d_Nreg, \
                self.tau, halfStep, self.size).wait()

        super(CLVisc, self).__updateGlobalMem(halfStep)


    def __output(self, nstep):
        super(CLVisc, self).__output(nstep)

    def evolve( self, ntskip=10 ):
        '''The main loop of hydrodynamic evolution '''
        for n in range(1000):
            self.__output(n)
            self.__stepUpdate(halfStep=np.int32(1))
            self.__stepUpdate(halfStep=np.int32(0))
            self.__updateGlobalMem()
            self.tau = cfg.real(cfg.TAU0 + (n+1)*cfg.DT)
            print 'EdMax= ',self.__edMax()
 


if __name__ == '__main__':
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print >>sys.stdout, 'start ...'
    t0 = time()
    visc = CLVisc()
    visc.initHydro()
    visc.evolve()
    t1 = time()
    print >>sys.stdout, 'finished. Total time: {dtime}'.format( dtime = t1-t0 )

