#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu May 12 07:10:22 2016

'''This script is used to test whether the HelmholtzHodge2D
decomposition work correctly or has a sign problem,
using hand make initial fluid vector'''

import matplotlib.pyplot as plt
import numpy as np

from HelmholtzHodge import HelmholtzHodge2D


def test_quiver1():
    '''for y = (1, 0), U[3,3] meas the upper left point'''
    x = np.linspace(0,1,11)
    y = np.linspace(1,0,11)
    u = v = np.zeros((11,11))
    u[3,3] = 0.2

    plt.quiver(x, y, u, v, scale=1)
    plt.show()


def test_quiver2():
    '''for y = (0, 1), U[3,3] meas the lower left point'''
    x = np.linspace(0,1,11)
    y = np.linspace(0,1,11)
    u = v = np.zeros((11,11))
    u[3,3] = 0.2

    plt.quiver(x, y, u, v, scale=1)
    plt.show()

def test_quiver3():
    x = np.linspace(0, 2*np.pi, 11)
    y = np.linspace(0, np.pi/2, 11)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = np.cos(X)
    V = np.sin(Y)

    plt.quiver(X, Y, U, V, scale=100)
    plt.show()

def test_quiver4():
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U = np.gradient(X)[0]
    V = np.gradient(Y)[1]

    ed = np.ones_like(U)

    U = -1 - np.cos(X**2 + Y)
    V = 1 + X - Y

    hh = HelmholtzHodge2D(U, V, x, y, ed)

    hh.quiver(scale=80)
    #hh.gradient_free(scale=10, color='r')
    hh.curl_free(scale=1000, color='b')
    plt.show()

def test_quiver5(kind='gradient_free'):
    x = np.linspace(-3, 3, 15)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ed = np.exp(-(X*X + Y*Y)/4 )

    grad_ed = np.gradient(ed)

    U, V = np.ones_like(ed), np.ones_like(ed)

    U += grad_ed[1]
    V += -grad_ed[0]

    U += -grad_ed[0]
    V += -grad_ed[1]

    hh = HelmholtzHodge2D(U, V, x, y, ed)

    #hh.quiver(scale=30)

    if kind == 'gradient_free':
        hh.gradient_free(scale=2, color='r')
    else:
        hh.curl_free(scale=50, color='b')

    plt.show()

test_quiver5(kind='curl_free')
#test_quiver5(kind='gradient_free')
