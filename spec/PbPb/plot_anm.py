#/usr/bin/env python
#auth r: lgpang
#email: lgpang@qq.com
#createTime: Tue 15 Sep 2015 13:55:52 CEST

import matplotlib.pyplot as plt
import numpy as np
import os
import math
#from common_plotting import smash_style


from matplotlib.lines import Line2D

def main():
    cent = ['0_5', '5_10', '10_20', '20_30', '30_40']
    line_styles = ['o', 's', 'd', '^', '*']
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    ymax = 0.0
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass


    for i, c in enumerate(cent):
        ann = np.loadtxt('anm_%s.dat'%c)
        plt.plot(np.arange(0, 11), ann, line_styles[i], color=colors[i], label=c)
        ymax = max(ymax, ann.max()) 

    plt.xticks(np.arange(0,11), (r'$\sqrt{<a_1^2>}$',
                               r'$\sqrt{<a_2^2>}$',
                               r'$\sqrt{<a_3^2>}$',
                               r'$\sqrt{<a_4^2>}$',
                               r'$\sqrt{<a_5^2>}$',
                               r'$\sqrt{<a_6^2>}$',
                               r'$\sqrt{-<a_1 a_3>}$',
                               r'$\sqrt{-<a_2 a_4>}$',
                               r'$\sqrt{-<a_3 a_5>}$',
                               r'$\sqrt{-<a_4 a_6>}$',
                               r'$\sqrt{-<a_5 a_7>}$'))
    plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV$')
    plt.xlim(-1, 11)
    plt.ylim(0, 1.2*ymax)

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    #smash_style.set()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
