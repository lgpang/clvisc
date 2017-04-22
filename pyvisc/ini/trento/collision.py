#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 21 Apr 2017 01:27:02 AM CEST

from subprocess import call
import pandas as pd
import os

__cwd__, __cwf__ = os.path.split(__file__)

class Collision(object):
    def __init__(self, centrality_file):
        self.info = pd.read_csv(centrality_file)

    def get_smin_smax(self, cent='0_6'):
        '''get min/max initial total entropy for one
        centrality class, stored in auau200.csv or ...'''
        self.info.set_index(['cent'])
        dat = self.info.loc[self.info['cent'] == cent]
        smin = dat['entropy_low'].values[0]
        smax = dat['entropy_high'].values[0]
        return smin, smax

class AuAu200(object):
    def __init__(self, cent, grid_max=15.0, grid_step=0.1):
        self.cross_section = 4.23
        self.projectile = 'Au'
        self.target = 'Au'
        self.grid_max = grid_max
        self.grid_step = grid_step
        centrality_file = os.path.join(__cwd__, 'auau200_cent.csv')
        coll = Collision(centrality_file)
        self.smin, self.smax = coll.get_smin_smax(cent)

    def create_ini(self, output_path):
        call(['trento', self.projectile, self.target, '1',
              '-o', output_path,
              '-x', '%s'%self.cross_section,
              '--s-min', '%s'%self.smin,
              '--s-max', '%s'%self.smax,
              '--grid-max', '%s'%self.grid_max,
              '--grid-step', '%s'%self.grid_step])

class PbPb2760(object):
    def __init__(self, cent, grid_max=15.0, grid_step=0.1):
        self.cross_section = 6.4
        self.projectile = 'Pb'
        self.target = 'Pb'
        self.grid_max = grid_max
        self.grid_step = grid_step
        centrality_file = os.path.join(__cwd__, 'pbpb2760_cent.csv')
        coll = Collision(centrality_file)
        self.smin, self.smax = coll.get_smin_smax(cent)

    def create_ini(self, output_path):
        call(['trento', self.projectile, self.target, '1',
              '-o', output_path,
              '-x', '%s'%self.cross_section,
              '--s-min', '%s'%self.smin,
              '--s-max', '%s'%self.smax,
              '--grid-max', '%s'%self.grid_max,
              '--grid-step', '%s'%self.grid_step])

class PbPb5020(object):
    def __init__(self, cent, grid_max=15.0, grid_step=0.1):
        self.cross_section = 7.0
        self.projectile = 'Pb'
        self.target = 'Pb'
        self.grid_max = grid_max
        self.grid_step = grid_step
        centrality_file = os.path.join(__cwd__, 'pbpb5020_cent.csv')
        coll = Collision(centrality_file)
        self.smin, self.smax = coll.get_smin_smax(cent)

    def create_ini(self, output_path):
        call(['trento', self.projectile, self.target, '1',
              '-o', output_path,
              '-x', '%s'%self.cross_section,
              '--s-min', '%s'%self.smin,
              '--s-max', '%s'%self.smax,
              '--grid-max', '%s'%self.grid_max,
              '--grid-step', '%s'%self.grid_step])


if __name__=='__main__':
    auau200 = AuAu200('0_6', grid_max=15.0, grid_step=0.1)
    auau200.create_ini('./dat')
