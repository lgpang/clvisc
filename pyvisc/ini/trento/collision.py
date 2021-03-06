#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 21 Apr 2017 01:27:02 AM CEST

from subprocess import call
import pandas as pd
import os
import numpy as np
import reader
from rotate import rotate

__cwd__, __cwf__ = os.path.split(__file__)

class Collision(object):
    def __init__(self, config):
        self.config = config
        centrality_file = os.path.join(__cwd__, config['centrality_file'])
        self.info = pd.read_csv(centrality_file)

    def get_smin_smax(self, cent='0_6'):
        '''get min/max initial total entropy for one
        centrality class, stored in auau200.csv or ...'''
        clow, chigh = cent.split('_')
        smin = self.entropy_bound(cent_bound = float(chigh))
        smax = self.entropy_bound(cent_bound = float(clow))
        return smin, smax

    def entropy_bound(self, cent_bound=5):
        '''get entropy value for one specific centrality bound'''
        self.info.set_index(['cent'])
        cents = self.info['cent']
        entropy = self.info['entropy']
        return np.interp(cent_bound, cents, entropy)

    def create_ini(self, cent, output_path,
                   grid_max=15.0, grid_step=0.1, num_of_events=1,
                   one_shot_ini=False, align_for_oneshot=False):
        smin, smax = self.get_smin_smax(cent)
        call(['trento', self.config['projectile'],
              self.config['target'],
              '%s'%num_of_events,
              '-o', output_path,
              '-x', '%s'%self.config['cross_section'],
              '--s-min', '%s'%smin,
              '--s-max', '%s'%smax,
              '--grid-max', '%s'%grid_max,
              '--grid-step', '%s'%grid_step])

        if one_shot_ini:
            ngrid = int(2 * grid_max / grid_step)
            sxy = np.zeros((ngrid, ngrid), dtype=np.float32)
            events = os.listdir(output_path)
            print(events)
            num_of_events = 0
            for event in events:
                try:
                    fname = os.path.join(output_path, event)
                    dat = np.loadtxt(fname).reshape(ngrid, ngrid)
                    opt = reader.get_comments(fname)
                    sd_new = rotate(dat, opt['ixcm'], opt['iycm'], opt['phi_2'], ngrid, ngrid)
                    sxy += sd_new 
                    num_of_events += 1
                except:
                    print(fname, 'is not a trento event')
            np.savetxt(os.path.join(output_path, "one_shot_ini.dat"), sxy/num_of_events, header=cent)


class AuAu200(Collision):
    def __init__(self):
        config = {'projectile':'Au',
                  'target':'Au',
                  'cross_section':4.23,
                  'centrality_file':'auau200_cent.csv'}
        super(AuAu200, self).__init__(config)

       

class PbPb2760(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':6.4,
                  'centrality_file':'pbpb2760_cent.csv'}
        super(PbPb2760, self).__init__(config)

       

class PbPb5020(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':7.0,
                  'centrality_file':'pbpb5020_cent.csv'}
        super(PbPb5020, self).__init__(config)

class XeXe5440(Collision):
    def __init__(self):
        config = {'projectile':'Xe',
                  'target':'Xe',
                  'cross_section':7.1,
                  'centrality_file':'xexe5440_cent.csv'}
        super(XeXe5440, self).__init__(config)


if __name__=='__main__':
    xexe = XeXe5440()
    xexe.create_ini('0_6', './dat', num_of_events=100, one_shot_ini=True)
