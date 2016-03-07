#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST
from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time

class Smearing(object):
    '''The pyopencl version for gaussian smearing ini condition'''
    def __init__(self, cfg, ctx, queue, compile_options, d_ev1,
        fname_partons, eos_table, SIGR=0.6, SIGZ=0.6, KFACTOR=1.0):
        '''initialize d_ev1 with partons p4x4 given by fname_partons'''
        self.cwd, cwf = os.path.split(__file__)
        self.gpu_defines = compile_options
        self.__loadAndBuildCLPrg(ctx, cfg, SIGR, SIGZ, KFACTOR)
        size = cfg.NX*cfg.NY*cfg.NZ
        h_p4x4 = np.zeros((size, 8), cfg.real)
        dat = np.loadtxt(fname_partons, skiprows=1, dtype=cfg.real)
        npartons = len(dat[:,0])
        h_p4x4 = dat.astype(cfg.real)
        print('num_of_partons=', npartons)
        d_p4x4 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_p4x4.nbytes)
        cl.enqueue_copy(queue, d_p4x4, h_p4x4)

        self.prg.smearing(queue, (cfg.NX, cfg.NY, cfg.NZ), (5,5,5),
                d_ev1, d_p4x4, eos_table, np.int32(npartons), np.int32(size)).wait()

    def __loadAndBuildCLPrg(self, ctx, cfg, SIGR, SIGZ, KFACTOR):
        #load and build *.cl programs with compile self.gpu_defines
        glauber_defines = list(self.gpu_defines)
        glauber_defines.append('-D {key}={value}f'.format(key='SQRTS', value=cfg.SQRTS))
        glauber_defines.append('-D {key}={value}f'.format(key='SIGR', value=SIGR))
        glauber_defines.append('-D {key}={value}f'.format(key='SIGZ', value=SIGZ))
        glauber_defines.append('-D {key}={value}f'.format(key='KFACTOR', value=KFACTOR))
        print(glauber_defines)
        with open(os.path.join(self.cwd, 'kernel', 'kernel_gaussian_smearing.cl'), 'r') as f:
            prg_src = f.read()
            self.prg = cl.Program(ctx, prg_src).build(
                                             options=glauber_defines)



class SmearingP4X4(object):
    '''The pyopencl version for gaussian smearing ini condition'''
    def __init__(self, cfg, ctx, queue, compile_options, d_ev1,
        p4x4, eos_table, SIGR=0.6, SIGZ=0.6, KFACTOR=1.0, force_bjorken=False):
        '''initialize d_ev1 with partons p4x4, which is one size*8 np.array '''
        self.cwd, cwf = os.path.split(__file__)
        self.gpu_defines = compile_options

        self.__loadAndBuildCLPrg(ctx, cfg, SIGR, SIGZ, KFACTOR)
        size = cfg.NX*cfg.NY*cfg.NZ
        h_p4x4 = np.zeros((size, 8), cfg.real)
        # read p4x4 from h5py for event_id
        dat = p4x4
        npartons = len(dat[:,0])
        h_p4x4 = dat.astype(cfg.real)
        print('num_of_partons=', npartons)
        d_p4x4 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_p4x4.nbytes)
        cl.enqueue_copy(queue, d_p4x4, h_p4x4)

        self.prg.smearing(queue, (cfg.NX, cfg.NY, cfg.NZ), (5,5,5),
                d_ev1, d_p4x4, eos_table, np.int32(npartons), np.int32(size)).wait()

        if force_bjorken:
            self.prg.force_bjorken(queue, (cfg.NX, cfg.NY, cfg.NZ), None,
                d_ev1, np.int32(size)).wait()


    def __loadAndBuildCLPrg(self, ctx, cfg, SIGR, SIGZ, KFACTOR):
        #load and build *.cl programs with compile self.gpu_defines
        glauber_defines = list(self.gpu_defines)
        glauber_defines.append('-D {key}={value}f'.format(key='SQRTS', value=cfg.SQRTS))
        glauber_defines.append('-D {key}={value}f'.format(key='SIGR', value=SIGR))
        glauber_defines.append('-D {key}={value}f'.format(key='SIGZ', value=SIGZ))
        glauber_defines.append('-D {key}={value}f'.format(key='KFACTOR', value=KFACTOR))
        print(glauber_defines)
        with open(os.path.join(self.cwd, 'kernel', 'kernel_gaussian_smearing.cl'), 'r') as f:
            prg_src = f.read()
            self.prg = cl.Program(ctx, prg_src).build(
                                             options=glauber_defines)




def main():
    '''set default platform and device in opencl'''
    #os.environ[ 'PYOPENCL_CTX' ] = '0:0'
    print('start ...')
    t0 = time()
    from config import cfg
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 61

    cfg.DT = 0.005
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.IEOS = 2
    cfg.ntskip = 40

    cfg.TAU0 = 0.2

    cfg.ETAOS = 0.08

    from visc import CLVisc
    visc = CLVisc(cfg)

    fname_partons = '/u/lpang/P10.txt'
    #fname_partons = '/data01/hyihp/pang/GammaJet/AuAu200_0_80/P1.txt'

    Smearing(cfg, visc.ctx, visc.queue, visc.compile_options,
            visc.ideal.d_ev[1], fname_partons, visc.eos_table)

    visc.evolve(max_loops=2400, save_hypersf=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))



def ideal_main():
    '''set default platform and device in opencl'''
    print('start ...')
    t0 = time()
    from config import cfg
    from ideal import CLIdeal
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 61

    cfg.DT = 0.02
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.IEOS = 2
    cfg.ntskip = 60

    cfg.TAU0 = 0.2

    cfg.ETAOS = 0.08

    ideal = CLIdeal(cfg)
    #fname_partons = '/data01/hyihp/pang/GammaJet/AuAu200_0_80/P1.txt'
    fname_partons = '/u/lpang/P10.txt'

    Smearing(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines,
            ideal.d_ev[1], fname_partons, ideal.eos_table)

    ideal.evolve(max_loops=2400)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))



if __name__ == '__main__':
    #ideal_main()
    main()
