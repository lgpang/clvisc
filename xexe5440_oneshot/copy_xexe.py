#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Mon 08 Aug 2016 12:10:38 PM CEST

import matplotlib.pyplot as plt
import numpy as np

from create_table import create_table_for_jet

from subprocess import call

import os

system = 'xexe5440'

centralities = ['0_1', '0_5', '5_10', '0_10', '10_30', '30_50', '50_70', '70_90', '0_80']

fbase = '/lustre/nyx/hyihp/lpang/xexe5440_oneshot/'

def create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

def copy_spec(src, dest):
    files = ['dN_over_2pidEtaptdpt_mc_charged.dat', 'dN_over_2pidYptdpt_mc_211.dat',
             'dN_over_2pidYptdpt_mc_321.dat', 'dN_over_2pidYptdpt_mc_2212.dat', 'bulk.dat',
             'vn_211.dat', 'vn_321.dat', 'vn_2212.dat', 'dNdEta_mc_charged.dat',
             'hydro.info']

    create_table_for_jet(src)
    print('create bulk.dat finished for', src)
    for f in files:
        fname_src = os.path.join(src, f)
        call(['cp', fname_src, dest])


def copy_to(dest):
    for cent in centralities:
        f_src = os.path.join(fbase, cent) 
        f_des = os.path.join(dest, cent)
        create_dir(f_des)
        copy_spec(f_src, f_des)
        call(['cp', os.path.join(f_src, 'trento_ini/one_shot_ini.dat'), dest])
        print(cent, ' finished!')


if __name__=='__main__':
    fdest = '/u/lpang/%s'%system
    if not os.path.exists(fdest):
        os.makedirs(fdest)

    copy_to(fdest)
