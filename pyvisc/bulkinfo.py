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
from eos.eos import Eos
#import matplotlib.pyplot as plt


class BulkInfo(object):
    '''The bulk information like:
       ed(x), ed(y), ed(eta), T(x), T(y), T(eta)
       vx, vy, veta, ecc_x, ecc_p'''
    def __init__(self, cfg, ctx, queue, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)

        self.ex, self.ey, self.ez = [], [], []
        self.vx, self.vy, self.vz = [], [], []
        self.exy, self.exz, self.vx_xy, self.vy_xy = [], [], [], []
        self.ecc_x = []
        self.ecc_p = []
        self.ecc2_vs_rapidity = []
        self.time = []
        self.edmax = []
        self.__loadAndBuildCLPrg()
        self.eos = Eos(cfg)


    def __loadAndBuildCLPrg(self):
        #load and build *.cl programs with compile self.gpu_defines
        edslice_src = '''#include"real_type.h"
            __kernel void get_ed(__global real4 * d_ev,
                                 __global real4 * d_ev_x0,
                                 __global real4 * d_ev_y0,
                                 __global real4 * d_ev_z0,
                                 __global real4 * d_ev_xy,
                                 __global real4 * d_ev_xz,
                                 __global real4 * d_ev_yz) {
                int gid = get_global_id(0);
                if ( gid < NX ) {
                    int j = NY/2; 
                    int k = NZ/2;
                    d_ev_x0[gid] = d_ev[gid*NY*NZ + j*NZ + k];

                    int i = gid;
                    for ( j = 0; j< NY; j ++ ) {
                        d_ev_xy[i*NY+j] = d_ev[i*NY*NZ + j*NZ + k];
                    }

                    j = NY/2;
                    for ( k = 0; k < NZ; k ++ ) {
                        d_ev_xz[i*NZ+k] = d_ev[i*NY*NZ + j*NZ + k];
                    }
                }

                if ( gid < NY ) {
                    int i = NX/2; 
                    int k = NZ/2;
                    d_ev_y0[gid] = d_ev[i*NY*NZ + gid*NZ + k];
                    int j = gid;
                    for ( k = 0; k < NZ; k ++ ) {
                        d_ev_yz[j*NZ+k] = d_ev[i*NY*NZ + j*NZ + k];
                    }
                }

                if ( gid < NZ ) {
                    int i = NX/2; 
                    int j = NY/2;
                    d_ev_z0[gid] = d_ev[i*NY*NZ + j*NZ + gid];
                }
            }
            '''
        self.kernel_edslice = cl.Program(self.ctx, edslice_src).build(
                                         options=self.compile_options)
    def get(self, tau, d_ev1, edmax):
        self.time.append(tau)
        self.edmax.append(edmax)
        mf = cl.mem_flags
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ

        self.ecc_vs_rapidity(d_ev1)

        h_ev1d = np.zeros((2000, 4), self.cfg.real)
        h_evxy = np.zeros((NX*NY, 4), self.cfg.real)
        h_evxz = np.zeros((NX*NZ, 4), self.cfg.real)
        h_evyz = np.zeros((NY*NZ, 4), self.cfg.real)

        d_evx0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)
        d_evy0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)
        d_evz0 = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_ev1d.nbytes)

        d_evxy = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evxy.nbytes)
        d_evxz = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evxz.nbytes)
        d_evyz = cl.Buffer(self.ctx, mf.READ_WRITE, size=h_evyz.nbytes)

        self.kernel_edslice.get_ed(self.queue, (2000,), None, d_ev1,
                d_evx0, d_evy0, d_evz0, d_evxy, d_evxz, d_evyz)

        h_evx0 = np.zeros((NX, 4), self.cfg.real)
        h_evy0 = np.zeros((NY, 4), self.cfg.real)
        h_evz0 = np.zeros((NZ, 4), self.cfg.real)
        cl.enqueue_copy(self.queue, h_evx0, d_evx0).wait()
        cl.enqueue_copy(self.queue, h_evy0, d_evy0).wait()
        cl.enqueue_copy(self.queue, h_evz0, d_evz0).wait()

        self.ex.append(h_evx0[:,0])
        self.ey.append(h_evy0[:,0])
        self.ez.append(h_evz0[:,0])

        self.vx.append(h_evx0[:,1])
        self.vy.append(h_evy0[:,2])
        self.vz.append(h_evz0[:,3])

        cl.enqueue_copy(self.queue, h_evxy, d_evxy).wait()
        cl.enqueue_copy(self.queue, h_evxz, d_evxz).wait()
        cl.enqueue_copy(self.queue, h_evyz, d_evyz).wait()

        self.exy.append(h_evxy[:,0].reshape(NX, NY))
        self.vx_xy.append(h_evxy[:,1].reshape(NX, NY))
        self.vy_xy.append(h_evxy[:,2].reshape(NX, NY))

    def eccp(self, ed, vx, vy, vz=0.0):
        ''' eccx = <y*y-x*x>/<y*y+x*x> where <> are averaged 
            eccp = <Txx-Tyy>/<Txx+Tyy> '''
        pre = self.eos.f_P(ed)
        u0 = 1.0/np.sqrt(1.0 - vx*vx - vy*vy - vz*vz)
        Tyy = (ed + pre)*u0*u0*vy*vy + pre
        Txx = (ed + pre)*u0*u0*vx*vx + pre
        return (Txx - Tyy).sum() / (Txx + Tyy).sum()


    def ecc_vs_rapidity(self, d_ev):
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        cl.enqueue_copy(self.queue, self.h_ev, d_ev).wait()
        bulk = self.h_ev.reshape(NX, NY, NZ, 4)
        eccp = np.empty(NZ)
        for k in range(NZ):
            ed = bulk[:,:,k,0]
            vx = bulk[:,:,k,1]
            vy = bulk[:,:,k,2]
            vz = bulk[:,:,k,3]
            eccp[k] = self.eccp(ed, vx, vy, vz)
        self.ecc2_vs_rapidity.append(eccp)
        
    def save(self):
        np.savetxt(self.cfg.fPathOut+'/ex.dat', np.array(self.ex).T)
        np.savetxt(self.cfg.fPathOut+'/ey.dat', np.array(self.ey).T)
        np.savetxt(self.cfg.fPathOut+'/ez.dat', np.array(self.ez).T)

        np.savetxt(self.cfg.fPathOut+'/Tx.dat', np.array(self.eos.f_T(self.ex)).T)
        np.savetxt(self.cfg.fPathOut+'/Ty.dat', np.array(self.eos.f_T(self.ey)).T)
        np.savetxt(self.cfg.fPathOut+'/Tz.dat', np.array(self.eos.f_T(self.ez)).T)

        np.savetxt(self.cfg.fPathOut+'/vx.dat', np.array(self.vx).T)
        np.savetxt(self.cfg.fPathOut+'/vy.dat', np.array(self.vy).T)
        np.savetxt(self.cfg.fPathOut+'/vz.dat', np.array(self.vz).T)

        if len(self.ecc2_vs_rapidity) != 0:
            np.savetxt(self.cfg.fPathOut+'/ecc2.dat',
                       np.array(self.ecc2_vs_rapidity).T)

        eccp = []
        for idx, exy in enumerate(self.exy):
            vx= self.vx_xy[idx]
            vy= self.vy_xy[idx]
            np.savetxt(self.cfg.fPathOut+'/edxy%d.dat'%idx, exy)
            np.savetxt(self.cfg.fPathOut+'/Txy%d.dat'%idx, self.eos.f_T(exy))
            np.savetxt(self.cfg.fPathOut+'/vx_xy%d.dat'%idx, vx)
            np.savetxt(self.cfg.fPathOut+'/vy_xy%d.dat'%idx, vy)

            eccp.append(self.eccp(exy[:, self.cfg.NY/2],
                vx[:, self.cfg.NY/2], vy[:, self.cfg.NY/2]))
        
        np.savetxt(self.cfg.fPathOut + '/eccp.dat',
                   np.array(zip(self.time, eccp)))

        np.savetxt(self.cfg.fPathOut + '/Tmax.dat',
                   np.array(zip(self.time, self.eos.f_T(self.edmax))))
