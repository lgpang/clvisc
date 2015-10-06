#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 16 Sep 2015 01:38:27 PM CEST

import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    fnames = ['cn_b0_5.dat', 'cn_b5_10.dat', 'cn_b20_30.dat', 'cn_b30_40.dat']
    cn = [np.loadtxt(fname) for fname in fnames]
    fig, ax = plt.subplots(2,2)
    x, y = np.meshgrid(np.arange(30),np.arange(30))

    ax[0, 0].plot_wireframe(cn[0])
    ax[0, 1].plot_wireframe(cn[1])
    ax[1, 0].plot_wireframe(cn[2])
    ax[1, 1].plot_wireframe(cn[3])

    plt.show()


