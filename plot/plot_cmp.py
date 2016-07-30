#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Tue 26 Jul 2016 02:14:09 PM CEST

import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    figure = plt.figure(figsize=(20, 12))
    plt.subplot(121)
    import plot_Txtau

    plt.xlabel(r'$r\ [fm]$', fontsize=40)
    plt.ylabel(r'$\tau\ [fm]$', fontsize=45)

    plt.text(-5, 4, "Pure Glue")
    plt.text(-5, 15, "Mixed Phase")
    plt.text(5, 28, "Glueball Gas")

    plt.subplot(122)
    import plot_Txtau_QCD
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.9, wspace=0.05)
    plt.xlabel(r'$r\ [fm]$')
    plt.text(19, 31, r'$T\ [GeV]$', fontsize=30)

    plt.text(-2, 4, "QGP")
    plt.text(-5, 10, "Crossover")
    plt.text(-2, 18, "HRG")

    plot_Txtau_QCD.smash_style.set(minorticks_on=False)

    cbar_ax = figure.add_axes([0.92, 0.15, 0.02, 0.75])
    plt.colorbar(plot_Txtau_QCD.CI, cax=cbar_ax)

    plt.savefig("su3_vs_qcd_with_tags.pdf")

    plt.show()

