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
        glauber_defines.append('-D {key}={value}f'.format(key='NumOfNucleons', value=cfg.A))
        glauber_defines.append('-D {key}={value}f'.format(key='Ro0', value=cfg.NucleonDensity))
        glauber_defines.append('-D {key}={value}f'.format(key='R', value=cfg.Ra))
        glauber_defines.append('-D {key}={value}f'.format(key='Eta', value=cfg.Eta))
        glauber_defines.append('-D {key}={value}f'.format(key='Si0', value=cfg.Si0))
        glauber_defines.append('-D {key}={value}f'.format(key='ImpactParameter', value=cfg.ImpactParameter))
        glauber_defines.append('-D {key}={value}f'.format(key='Edmax', value=cfg.Edmax))
        glauber_defines.append('-D {key}={value}f'.format(key='Hwn', value=cfg.Hwn))
        glauber_defines.append('-D {key}={value}f'.format(key='Eta_flat', value=cfg.Eta_flat))
        glauber_defines.append('-D {key}={value}f'.format(key='Eta_gw', value=cfg.Eta_gw))
        print(glauber_defines)
        with open(os.path.join(self.cwd, 'kernel', 'kernel_glauber.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_glauber = cl.Program(ctx, prg_src).build(
                                             options=glauber_defines)

