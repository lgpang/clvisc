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


def weight_mean_b(cent_low, cent_high, system='Au+Au'):
    '''calc <b> for centrality class [cent_low, cent_high] for Au+Au
       200 GeV collisions and Pb+Pb 2.76 TeV'''
    from scipy.interpolate import interp1d
    from scipy.integrate import quad
    cent, nwn, bimp = None, None, None

    if system == 'Au+Au':
        # centrality classes
        cent = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90])
        # number of wounded nucleons
        nwn = np.array([393.0, 327.0, 278.0, 237.0, 202.0,  144.0,  99.0,  65.0,
               44.0, 22.0,  11.0, 4.0])
        # impact parameter ranges
        bimp = np.array([0.0, 3.3, 4.7, 5.8, 6.7, 8.2, 9.4, 10.6, 11.6, 12.5, 13.4, 14.3])
    elif system == 'Pb+Pb':
         # centrality classes
        cent = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90])
        # number of wounded nucleons
        nwn = np.array([416.0, 357.0, 305.0, 261.0, 216.0,  159.0,  110.0,  70.0,
               42.0, 23.0,  11.0, 4.0])
        # impact parameter ranges
        bimp = np.array([0.0, 3.54, 5.00, 6.13, 7.08, 8.67, 10.01, 11.19, 12.26, 13.24, 14.16, 15.07])
    else:
        print('collisions system must be Au+Au or Pb+Pb')
        exit(0)
   
    # impact_parameter as a function of centrality
    bimp_vs_cent = interp1d(cent, bimp, kind='cubic')
    # number of events as a functions of centrality
    nevents_vs_cent = interp1d(cent, bimp*bimp, kind='cubic')

    # number of wounded nucleon as a function of centrality
    nwn_vs_cent = interp1d(cent, nwn, kind='cubic')
    
    nevent = lambda b: b*b
    nwn_vs_b = interp1d(bimp, nwn, kind='cubic')
    
    #f1 = lambda c: bimp_vs_cent(c)*nevents_vs_cent(c)*nwn_vs_cent(c)
    #f2 = lambda c: nevents_vs_cent(c)*nwn_vs_cent(c)
    f1 = lambda b: b*nevent(b)*nwn_vs_b(b)
    f2 = lambda b: nevent(b)*nwn_vs_b(b)
    
    bmin = bimp_vs_cent(cent_low)
    bmax = bimp_vs_cent(cent_high)
    return quad(f1, bmin, bmax)[0] / quad(f2, bmin, bmax)[0]


class Glauber(object):
    '''The pyopencl version for glauber ini condition'''
    def __init__(self, cfg, ctx, queue, compile_options, d_ev1):
        '''Def hydro in opencl with params stored in self.__dict__ '''
        # create opencl environment
        self.cwd, cwf = os.path.split(__file__)
        self.gpu_defines = compile_options
        self.__loadAndBuildCLPrg(ctx, cfg)
        self.kernel_glauber.glauber_ini(queue, (cfg.NX, cfg.NY), None,
                                        d_ev1).wait()

    def __loadAndBuildCLPrg(self, ctx, cfg):
        #load and build *.cl programs with compile self.gpu_defines
        glauber_defines = list(self.gpu_defines)
        glauber_defines.append('-D {key}=(real){value}'.format(key='NumOfNucleons', value=cfg.A))
        glauber_defines.append('-D {key}=(real){value}'.format(key='SQRTS', value=cfg.SQRTS))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Ro0', value=cfg.NucleonDensity))
        glauber_defines.append('-D {key}=(real){value}'.format(key='R', value=cfg.Ra))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Eta', value=cfg.Eta))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Si0', value=cfg.Si0))
        glauber_defines.append('-D {key}=(real){value}'.format(key='ImpactParameter', value=cfg.ImpactParameter))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Edmax', value=cfg.Edmax))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Hwn', value=cfg.Hwn))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Eta_flat', value=cfg.Eta_flat))
        glauber_defines.append('-D {key}=(real){value}'.format(key='Eta_gw', value=cfg.Eta_gw))
        print(glauber_defines)
        with open(os.path.join(self.cwd, 'kernel', 'kernel_glauber.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_glauber = cl.Program(ctx, prg_src).build(
                                             options=glauber_defines)

    def save_nbinary(self, ctx, queue, cfg, dx=0.3, dy=0.3,
                     xlow = -9.75, ylow = -9.75, nx=66, ny=66):
        h_nbin = np.empty(nx*ny, cfg.real)
        mf = cl.mem_flags
        d_nbin = cl.Buffer(ctx, mf.READ_WRITE, size=h_nbin.nbytes)
        self.kernel_glauber.num_of_binary_collisions(queue, (nx, ny, ), None,
                d_nbin, cfg.real(xlow), cfg.real(ylow),
                cfg.real(dx), cfg.real(dy), np.int32(nx), np.int32(ny)).wait()
        cl.enqueue_copy(queue, h_nbin, d_nbin).wait()
        np.savetxt(cfg.fPathOut+'/nbin.dat', h_nbin.reshape(nx, ny))



if __name__ == '__main__':
    cent_min = [0, 5, 0, 10, 20, 10, 30, 50, 0]
    cent_max = [5, 10,10,15, 30, 30, 50, 80, 80]

    system = 'Au+Au'
    print('Au+Au 200 GeV')
    for i in range(len(cent_min)):
        cmin = cent_min[i]
        cmax = cent_max[i]
        print(cmin, cmax, weight_mean_b(cmin, cmax, 'Au+Au'))

    print('Pb+Pb 2760 GeV')
    for i in range(len(cent_min)):
        cmin = cent_min[i]
        cmax = cent_max[i]
        print(cmin, cmax, weight_mean_b(cmin, cmax, 'Pb+Pb'))
    

