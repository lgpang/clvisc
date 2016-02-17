#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Mi 17 Feb 2016 16:18:51 CET
''' calc the magnetic reponse of the QGP
with fluid velocity given by hydrodynamic simulations'''

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
#import pyopencl as cl


class MagneticField(object):
    def __init__(self, eB0, sigx, sigy, hydro_dir):
        '''eB0: maximum magnetic field
           sigx: gaussian width of magnetic field along x
           sigy: gaussian width of magnetic field along y
           hydro_dir: directory with fluid velocity profile
        '''
        self.eB0 = eB0
        self.sigx = sigx
        self.sigy = sigy
        self.hydro_dir = hydro_dir

    def B0(self, x, y):
        '''parametrization of magnetic field in the transverse plane
        with a gaussian distribution '''
        By = self.eB0 * np.exp(-x*x/(2*self.sigx*self.sigx)
                -y*y/(2*self.sigy*self.sigy))
        Bx = np.zeros_like(By)
        Bz = np.zeros_like(By)
        return np.array(Bx, By, Bz)

    def E(self, v, B):
        ''' E = - v cross B
        Notice we need nabla_z v_x, nabla_z v_y later'''
        Ex = v[:,1]*B[:,2] - v[:,2]*B[:,1]
        Ex = v[:,1]*B[:,2] - v[:,2]*B[:,1]


if __name__=='__main__':


