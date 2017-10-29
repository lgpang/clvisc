#!/usr/bin/env python
# -*- utf8 -*-
'''PbPb 2.76 TeV collisions data '''
import os
import numpy as np
import pandas as pd
from load import HepData, json_from_file
import matplotlib.pyplot as plt
import json

class dNdEta(object):
    def __init__(self):
        url = "https://www.hepdata.net/record/data/68753/60859/1"
        saved_fname = "data/pbpb2760_dndeta.json"
        if not os.path.exists(saved_fname):
            self.json_data = HepData(url).json_data
            with open(saved_fname, 'w') as fout:
                json.dump(self.json_data, fout, sort_keys=True,
                          indent=2)
        else:
            self.json_data = json_from_file(saved_fname)

        # x is eta
        self.x = None
        # y is dndeta
        self.y = {}
        self.yerr = {}
        self.__parse(self.json_data)

    def address(self):
        '''url address for the data page '''
        print("https://hepdata.net/record/ins1225979")

    def cite(self):
        '''inspire id for citation '''
        print("\cite{Abbas:2013bpa}")

    def bibtex(self):
        print('''
            @article{Abbas:2013bpa,
                  author         = "Abbas, Ehab and others",
                  title          = "{Centrality dependence of the pseudorapidity density
                                    distribution for charged particles in Pb-Pb collisions at
                                    $\sqrt{s_{\rm NN}}$ = 2.76 TeV}",
                  collaboration  = "ALICE",
                  journal        = "Phys. Lett.",
                  volume         = "B726",
                  year           = "2013",
                  pages          = "610-622",
                  doi            = "10.1016/j.physletb.2013.09.022",
                  eprint         = "1304.0347",
                  archivePrefix  = "arXiv",
                  primaryClass   = "nucl-ex",
                  reportNumber   = "CERN-PH-EP-2013-045",
                  SLACcitation   = "%%CITATION = ARXIV:1304.0347;%%"
            }''')

    def __parse(self, data):
        '''get pseudo-rapidity list'''
        self.x = np.array([0.5*(p['x'][0]['high'] + p['x'][0]['low']) for p in data['values']])
        self.y['0-5'] = np.array([p['y'][0]['value'] for p in data['values']])
        self.y['5-10'] = np.array([p['y'][1]['value'] for p in data['values']])
        self.y['10-20'] = np.array([p['y'][2]['value'] for p in data['values']])
        self.y['20-30'] = np.array([p['y'][3]['value'] for p in data['values']])

        self.yerr['0-5']   = np.array([p['y'][0]['errors'][0]['symerror'] for p in data['values']])
        self.yerr['5-10']  = np.array([p['y'][1]['errors'][0]['symerror'] for p in data['values']])
        self.yerr['10-20'] = np.array([p['y'][2]['errors'][0]['symerror'] for p in data['values']])
        self.yerr['20-30'] = np.array([p['y'][3]['errors'][0]['symerror'] for p in data['values']])

    def plot(self):
        cents = ['0-5', '5-10', '10-20', '20-30']
        for cent in cents:
            plt.errorbar(self.x, self.y[cent], self.yerr[cent])
        plt.show()


class dNdPt(object):
    def __init__(self):
        url = "https://hepdata.net/record/ins1377750?format=json"
        saved_fname = "data/pbpb2760_dndpt.json"
        if not os.path.exists(saved_fname):
            self.json_data = HepData(url).json_data
            with open(saved_fname, 'w') as fout:
                json.dump(self.json_data, fout, sort_keys=True,
                          indent=2)
        else:
            self.json_data = json_from_file(saved_fname)


        #info = HepData(url).json_data
        info = self.json_data
        self.json = {}
        #### pion = pion+ + pion-, kaon = kaon+ + kaon-, proton = p+ + p-
        self.json['pion'] = HepData(info['data_tables'][0]['data']['json']).json_data
        self.json['kaon'] = HepData(info['data_tables'][1]['data']['json']).json_data
        self.json['proton'] = HepData(info['data_tables'][2]['data']['json']).json_data

        self.data = {}
        self.__parse('pion')
        self.__parse('kaon')
        self.__parse('proton')

    def address(self):
        '''url address for the data page '''
        print("https://hepdata.net/record/ins1377750")

    def cite(self):
        '''inspire id for citation '''
        print("\cite{Adam:2015kca}")

    def bibtex(self):
        print('''
            @article{Adam:2015kca,
                  author         = "Adam, Jaroslav and others",
                  title          = "{Centrality dependence of the nuclear modification factor
                                    of charged pions, kaons, and protons in Pb-Pb collisions
                                    at $\sqrt{s_{\rm NN}}=2.76$ TeV}",
                  collaboration  = "ALICE",
                  journal        = "Phys. Rev.",
                  volume         = "C93",
                  year           = "2016",
                  number         = "3",
                  pages          = "034913",
                  doi            = "10.1103/PhysRevC.93.034913",
                  eprint         = "1506.07287",
                  archivePrefix  = "arXiv",
                  primaryClass   = "nucl-ex",
                  reportNumber   = "CERN-PH-EP-2015-152",
                  SLACcitation   = "%%CITATION = ARXIV:1506.07287;%%"
            }''')

    def __rapidity_type(self, js):
        '''get rapidity type for (1/2pi) dN/dRap ptdpt
        return: 'YRAP' or 'ETARAP'
        '''
        return js["qualifiers"]['ETARAP'][0]['type']
        
    def __rapidity_range(self, js):
        '''get rapidity range which is used to compute (1/2pi) dN/dRap ptdpt
        return: 'YRAP' or 'ETARAP'
        '''
        rap = js["qualifiers"]['ETARAP'][0]['value'].split('-')
        return -float(rap[1]), float(rap[2])
        
 
    def __parse(self, particle_type='pion'):
        js = self.json[particle_type]
        self.data[particle_type] = {}
        self.data[particle_type]['x'] = np.array([0.5*(p['x'][0]['high'] + p['x'][0]['low']) for p in js['values']])
        self.data[particle_type]['y'] = {}
        self.data[particle_type]['yerr'] = {}
        cent = ['0-5', '5-10', '10-20', '20-40', '40-60', '60-80']
        self.data[particle_type]['cent'] = cent
        for idx, c in enumerate(cent):
            self.data[particle_type]['y'][c] = np.array([p['y'][idx]['value'] for p in js['values']])
            self.data[particle_type]['yerr'][c] = np.array([p['y'][idx]['errors'][1]['symerror'] for p in js['values']])

        self.data[particle_type]['rapidity_type'] = self.__rapidity_type(js)
        self.data[particle_type]['rapidity_range'] = self.__rapidity_range(js)

    def get(self, hadron='pion', cent='0-5'):
        pt = self.data[hadron]['x']
        spec = self.data[hadron]['y'][cent]
        yerr = self.data[hadron]['yerr'][cent]
        yerr_low = yerr_high = yerr
        return pt, spec, yerr_low, yerr_high

    def plot_cent(self, particle_type='pion'):
        import matplotlib.pyplot as plt
        hadron = self.data[particle_type]
        for cent in hadron['cent']:
            plt.errorbar(hadron['x'], hadron['y'][cent], hadron['yerr'][cent])
        plt.gca().set_yscale('log')
        plt.show()

    def plot_identify(self, cent='0-5'):
        import matplotlib.pyplot as plt
        hadrons = ['pion', 'kaon', 'proton']

        for h_type in hadrons:
            hadron = self.data[h_type]
            plt.errorbar(hadron['x'], hadron['y'][cent], hadron['yerr'][cent])

        plt.gca().set_yscale('log')
        plt.show()



############# START  PT differential and pt integrated vn ############

class Vn(object):
    def __init__(self, n=2):
        self.pt_diff = {}
        self.pt_intg = {}
        self.harmonic_order = n
        self.__parse()

    def __parse(self):
        cent = ['0-1', '0-5', '5-10', '10-20', '20-30', '30-40', '40-50']
        start_idx = 49 + 21 * (self.harmonic_order - 2)
        table_pt_diff = {'pion':{c:'Table%s.csv'%(start_idx+idx) for idx, c in enumerate(cent)},
                         'kaon':{c:'Table%s.csv'%(start_idx+7+idx) for idx, c in enumerate(cent)},
                         'proton':{c:'Table%s.csv'%(start_idx+14+idx) for idx, c in enumerate(cent)}}

        intg_start = 217 + 3 * (self.harmonic_order - 2)
        table_pt_intg = {'pion':'Table%s.csv'%(intg_start),
                         'kaon':'Table%s.csv'%(intg_start+1),
                         'proton':'Table%s.csv'%(intg_start+2)}

        base_path = os.path.join(os.getcwd(), 'data/pbpb2760_identified_v2_csv/')
        for hadron_type in table_pt_diff:
            vn_cent = {}
            for cent in table_pt_diff[hadron_type]:
                path = os.path.join(base_path, table_pt_diff[hadron_type][cent])
                vn_cent[cent] = pd.read_table(path, sep=',', comment='#')
            self.pt_diff[hadron_type] = vn_cent

        for hadron_type in table_pt_intg:
            path = os.path.join(base_path, table_pt_intg[hadron_type])
            self.pt_intg[hadron_type] = pd.read_table(path, sep=',', comment='#')

    def address(self):
        print("https://hepdata.net/record/ins900651")
        # pt diff v2 of charged h, proton from ep with deta>2
        print("https://hepdata.net/record/ins1116150")
        # v2 of charged particle from ep, c2, c4, lyz
        print("https://hepdata.net/record/ins1107659")

    def cite(self):
        '''print inspire id to cite the data'''
        print("\cite{Adam:2016nfo}")

    def bibtex(self):
        print('''
            @article{Adam:2016nfo,
            author         = "Adam, Jaroslav and others",
            title          = "{Higher harmonic flow coefficients of identified hadrons
                              in Pb-Pb collisions at $\sqrt{s_{\rm NN}}$ = 2.76 TeV}",
            collaboration  = "ALICE",
            journal        = "JHEP",
            volume         = "09",
            year           = "2016",
            pages          = "164",
            doi            = "10.1007/JHEP09(2016)164",
            eprint         = "1606.06057",
            archivePrefix  = "arXiv",
            primaryClass   = "nucl-ex",
            reportNumber   = "CERN-EP-2016-159",
            SLACcitation   = "%%CITATION = ARXIV:1606.06057;%%"
            }''')

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




if __name__ == '__main__':
    #dndpt.plot_identify('60-80')
    #print(dndpt.data['pion']['rapidity_range'])
    dndeta = dNdEta()
    print(dndeta.y['0-5'])
    dndpt = dNdPt()
    #dndeta.plot()
    v3 = Vn(n=3)
    v3.cite()
    v3.plot_ptdiff_vs_cent()
