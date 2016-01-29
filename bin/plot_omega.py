#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 29 Jan 2016 16:17:16 CET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__=='__main__':
    NX, NY, NZ = 301, 301, 61
    #dat = np.loadtxt('omegamu_60.dat')
    dat = pd.read_csv('omegamu_60.dat', sep=' ', header=None, skiprows=1).values
    omega_y = dat[:, 2].reshape(301, 301, 61)

    #plt.imshow(omega_y[:, 150, :], vmin=-1, vmax=1)
    plt.imshow(omega_y[:, :, 30], vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()



