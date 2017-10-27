#!/usr/bin/env python
# -*- utf8 -*-
'''vn as a function of pseudo-rapidity data'''
import os
import numpy as np
import pandas as pd
from load import HepData, json_from_file
import matplotlib.pyplot as plt
import json
try:
    # used in python 2.*
    from StringIO import StringIO as fstring
    #fstr = fstring(str(proc))
except ImportError:
    # used in python 3.*
    from io import StringIO as fstring
    #fstr = fstring(str(proc, 'utf-8'))



class Vn_vs_eta(object):
    def __init__(self):
        self.collision_system = None
        self.centralities = {}
        self.data = {}
        self.__parse()

    def __parse(self):
        pass
   
    def address(self):
        print("")

    def cite(self):
        '''print inspire id to cite the data'''
        print("")

    def bibtex(self):
        print("")

    def get_ptintg(self, hadron='pion'):
        dat = self.pt_intg[hadron].values
        cent, intg, err_m, err_p = dat[:,0], dat[:, 1], dat[:, 2], dat[:, 3]
        yerr_low = -err_m
        yerr_high = err_p
        return cent, intg, yerr_low, yerr_high

    def plot_ptintg(self, hadron='pion'):
        cent, intg, err0, err1 = self.get_ptintg(hadron)
        plt.errorbar(cent, intg, yerr=(err0, err1))
        plt.show()

    def get_ptdiff(self, hadron='pion', cent='0-5'):
        '''return pt, v2, v2_err_low, v2_err_high '''
        dat = self.pt_diff[hadron][cent].values
        pt, vn = dat[:, 0], dat[:, 1]
        yerr_low = - dat[:, 4]
        yerr_high = dat[:, 5]
        return pt, vn, yerr_low, yerr_high

    def plot_ptdiff_vs_cent(self, hadron='pion'):
        cent = {'0-5', '5-10', '10-20', '20-30', '30-40', '40-50'}
        for c in cent:
            x, y, yerr0, yerr1 = self.get_ptdiff(hadron, c)
            plt.errorbar(x, y, yerr=(yerr0, yerr1))
        plt.show()



class PbPb2760(Vn_vs_eta):
    def __init__(self):
        '''The pseudo-rapidity dependence of the anisotropic flows of charged particles
           in Pb+Pb 2.76 TeV collisions from ALICE collaboration'''
        self.collision_system = "pbpb2760"
        # self.info stores the line number for vn{2,4} in each Table
        self.info = {'0-5':{'table_id':1,   'files':['v22', 'v24', 'v32', 'v42']},
                     '5-10':{'table_id':2,  'files':['v22', 'v24', 'v32', 'v42']},
                     '10-20':{'table_id':3, 'files':['v22', 'v24', 'v32', 'v42']},
                     '20-30':{'table_id':4, 'files':['v22', 'v24', 'v32', 'v42']},
                     '30-40':{'table_id':5, 'files':['v22', 'v24', 'v32', 'v42']},
                     '40-50':{'table_id':6, 'files':['v22', 'v24', 'v32', 'v42']},
                     '50-60':{'table_id':7, 'files':['v22', 'v24', 'v32']},
                     '60-70':{'table_id':8, 'files':['v22', 'v24']},
                     '70-80':{'table_id':9, 'files':['v22', 'v24']}}

        self.data = {}
        self.__parse()

    def address(self):
        print("https://hepdata.net/record/ins1456145")

    def cite(self):
        '''print inspire id to cite the data'''
        print("\cite{Adam:2016ows}")

    def bibtex(self):
        print(''' @article{Adam:2016ows,
                  author         = "Adam, Jaroslav and others",
                  title          = "{Pseudorapidity dependence of the anisotropic flow of
                                    charged particles in Pb-Pb collisions at $\sqrt{s_{\rm
                                    NN}}=2.76$ TeV}",
                  collaboration  = "ALICE",
                  journal        = "Phys. Lett.",
                  volume         = "B762",
                  year           = "2016",
                  pages          = "376-388",
                  doi            = "10.1016/j.physletb.2016.07.017",
                  eprint         = "1605.02035",
                  archivePrefix  = "arXiv",
                  primaryClass   = "nucl-ex",
                  reportNumber   = "CERN-EP-2016-115",
                  SLACcitation   = "%%CITATION = ARXIV:1605.02035;%%"
            }''')

    def __parse(self):
        for cent, headers in self.info.items():
            idx = headers['table_id']
            base_path = os.path.join(os.getcwd(), 'data/ALICE_pbpb2760_vn_vs_eta/', 'Table%s.csv'%idx)
            cent_data = {}
            with open (base_path, 'r') as myfile:
                data = myfile.read().split('\n\n')[:-1]
                for idx, vn_type in enumerate(headers['files']):
                    data_str = fstring(data[idx])
                    cent_data[vn_type] = pd.read_csv(data_str, sep=',', comment='#')
            self.data[cent] = cent_data

    def get(self, cent, vn_type):
        data = self.data[cent][vn_type].values
        eta = data[:, 0]
        vn = data[:, 1]
        sta_err1 = -data[:, 3]
        sta_err2 = data[:, 2]
        sys_err1 = -data[:, 5]
        sys_err2 = data[:, 4]
        return eta, vn, sta_err1, sta_err2, sys_err1, sys_err2


if __name__ == '__main__':
    vn_vs_eta = PbPb2760()
    cents = ['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
    for cent in cents:
        data = vn_vs_eta.get(cent, 'v22')
        plt.errorbar(data[0], data[1], yerr=(data[4], data[5]), fmt='o')
    plt.show()
