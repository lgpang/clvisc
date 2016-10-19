#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 15 Oct 2015 14:02:44 CEST

from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)

class PimnInfo(object):
    '''The pimn information like:
       pi^{mu nu} along x, y, eta, xy, x-eta, y-eta
       traceless, transverse properties '''
    def __init__(self, cfg, ctx, queue, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        # pimn_x stores pi00, pi01, pi02, pi03, pi11, pi12, pi13
        # pi22, pi23, pi33 along x for (y=0, etas=0)
        self.pixx_x = []
        self.piyy_x = []
        self.pizz_x = []
        self.time = []
        self.__loadAndBuildCLPrg()

        self.x = np.linspace(-NX/2*cfg.DX, NX/2*cfg.DX, NX)
        self.y = np.linspace(-NY/2*cfg.DY, NY/2*cfg.DY, NY)
        self.z = np.linspace(-NZ/2*cfg.DZ, NZ/2*cfg.DZ, NZ)


    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile self.gpu_defines
        pislice_src = '''#include"real_type.h"
            __kernel void get(__global real * d_pi,
                                 __global real * d_pimn_x){
                int gid = get_global_id(0);
                if ( gid < NX ) {
                    int j = NY/2; 
                    int k = NZ/2;
                    int I = gid*NY*NZ + j*NZ + k;
                    for ( int mn = 0; mn < 10; mn++ ) {
                        d_pimn_x[10*gid + mn] = d_pi[10*I + mn];
                    }

                    //int i = gid;
                    //for ( j = 0; j< NY; j ++ ) {
                    //    d_ev_xy[i*NY+j] = d_ev[i*NY*NZ + j*NZ + k];
                    //}

                    //j = NY/2;
                    //for ( k = 0; k < NZ; k ++ ) {
                    //    d_ev_xz[i*NZ+k] = d_ev[i*NY*NZ + j*NZ + k];
                    //}
                }
            }
            '''
        self.kernel_pislice = cl.Program(self.ctx, pislice_src).build(
                                 options=' '.join(self.compile_options))
    def get(self, tau, d_pi):
        self.time.append(tau)
        mf = cl.mem_flags
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ

        h_pix  = np.zeros((NX, 10), self.cfg.real)
        h_pixy = np.zeros((NX*NY, 10), self.cfg.real)

        d_pix = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_pix.nbytes)
        d_pixy = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_pixy.nbytes)

        self.kernel_pislice.get(self.queue, (2000,), None, d_pi,
                d_pix)

        cl.enqueue_copy(self.queue, h_pix, d_pix).wait()

        pimn_x = h_pix.reshape(NX, 10)
        self.pixx_x.append(pimn_x[:, 4])
        self.piyy_x.append(pimn_x[:, 7])
        self.pizz_x.append(pimn_x[:, 9])

       
    def save(self):
        np.savetxt(self.cfg.fPathOut+'/pizz_x.dat', np.array(self.pizz_x).T)
        np.savetxt(self.cfg.fPathOut+'/pixx_x.dat', np.array(self.pixx_x).T)
        np.savetxt(self.cfg.fPathOut+'/piyy_x.dat', np.array(self.piyy_x).T)

