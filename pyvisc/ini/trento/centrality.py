#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sun 05 Jun 2016 10:12:20 PM CEST
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Centrality(object):
    def __init__(self, nch_list):
        self.non_bias = np.sort(nch_list)[::-1]
        self.num_of_events = len(self.non_bias)

    def get_centrality_class(self, cent_min, cent_max):
        '''get the nch range [nch_max, nch_min] for centrality class [cent_min, cent_max]
        which can be used later to select events for this centrality bin'''
        eid = lambda cent: int(self.num_of_events * cent * 0.01)
        nch_max = self.non_bias[eid(cent_min)]
        nch_min = self.non_bias[eid(cent_max)]
        return nch_max, nch_min

    def create_table(self, file_name):
        eid = lambda cent: int(self.num_of_events * cent * 0.01)
        centralities = []
        ini_total_entropy = []
        for c in range(101):
            idx = min(eid(c), self.num_of_events-1)
            s = self.non_bias[idx]
            centralities.append(c)
            ini_total_entropy.append(s)
        df = pd.DataFrame({'cent':centralities, 'entropy':ini_total_entropy})
        df.to_csv(file_name)

def dist(nch):
    plt.hist(nch, bins=50)
    #plt.yscale('log', nonposy='clip')

def e2_e3_scatter(ecc2, ecc3):
    plt.scatter(ecc2, ecc3)
    plt.show()

def Nch_vs_Npart(Nch, Npart):
    plt.scatter(Npart, Nch/(0.5*Npart))
    plt.show()



if __name__=='__main__':
    path = '/lustre/nyx/hyihp/lpang/trento_ini/'
    #dat = np.loadtxt(os.path.join(path, 'pbpb1million.log'))
    dat = np.loadtxt(os.path.join(path, 'xexe5440.log'))

    entropy = dat[:, 3]
    centrality = Centrality(entropy)
    centrality.create_table('xexe5440_cent.csv')


