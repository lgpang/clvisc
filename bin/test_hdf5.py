#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import matplotlib.pyplot as plt
import numpy as np
import h5py



def visit(h5py_file):
    def print_name(name):
        print(name)
    h5py_file.visit(print_name)


def read_p4x4(cent='30_35', idx=0,
        fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    '''read 4-momentum and 4-coordiantes from h5 file,
    return: np.array with shape (num_of_partons, 8)
    the first 4 columns store: E, px, py, pz
    the last 4 columns store: t, x, y, z'''
    with h5py.File(fname, 'r') as f:
        grp = f['cent']
        event_id = grp[cent][:, 0].astype(np.int)

        impact = grp[cent][:, 1]
        nw = grp[cent][:, 2]
        nparton = grp[cent][:, 3]
        key = 'event%s'%event_id[idx]
        #print key, nw[0], nparton[0]
        p4x4 = f[key]
        print('len(p4x4)=', len(p4x4))
        print('nparton_ampt=', nparton[idx])
        print('attrs[nparton]=', p4x4.attrs['num_of_partons_cross_tau0'])
        print(key, p4x4.attrs['event_id'])
        return p4x4[...], event_id[idx], impact[idx], nw[idx], nparton[idx]



def renew_num_partons(fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    with h5py.File(fname, 'r+') as f:
        old_np = f['glauber/np'][...]
        npartons = []
        for event_id in range(len(old_np)):
            dset = f['event%s'%event_id]
            print(event_id, dset.attrs['num_of_partons_cross_tau0'])
            npartons.append(dset.attrs['num_of_partons_cross_tau0'])

        del f['num_partons_p4x4']
        dset = f.create_dataset('num_partons_p4x4', (len(npartons), ), dtype = 'f')
        dset[...] = np.array(npartons)
            

def renew_centrality(fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    ''' for cent 0_5, 5_10, ... 90_95 '''
    with h5py.File(fname, 'r+') as f:
        num_partons = f['num_partons_p4x4'][...]
        non_bias= np.sort(num_partons)[::-1]

        bins = np.linspace(0, 95, 20, endpoint=True)
        print(bins)
        ids = (len(non_bias)*bins*0.01).astype(np.int)
        mul = non_bias[ids]
        for i in xrange(19):
            print 'multiplicity=({low},{high})'.format(low=mul[i], high=mul[i+1])
            cent = []
            for j, npartons in enumerate(num_partons):
                if npartons > mul[i+1] and npartons < mul[i]:
                    p4x4 = f['event%s'%j]
                    impb = p4x4.attrs['impact_parameter']
                    nwnd = p4x4.attrs['num_of_participants']
                    cent.append([j, impb, nwnd, npartons])
            cent_key = 'cent/%s_%s'%(int(bins[i]), int(bins[i+1]))
            #del f[cent_key]
            #event_ids = f.create_dataset(cent_key, (len(cent), 4))
            event_ids = f[cent_key]
            event_ids[...] = np.array(cent)
            event_ids.attrs['mul_high'] = mul[i]
            event_ids.attrs['mul_low'] = mul[i+1]


def renew_centrality_1(fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    ''' for cent 0_10, 10_20, ... 90_100 '''
    with h5py.File(fname, 'r+') as f:
        num_partons = f['num_partons_p4x4'][...]
        non_bias= np.sort(num_partons)[::-1]

        bins = np.linspace(0, 90, 10, endpoint=True)
        print(bins)
        ids = (len(non_bias)*bins*0.01).astype(np.int)
        mul = non_bias[ids]
        for i in xrange(9):
            print 'multiplicity=({low},{high})'.format(low=mul[i], high=mul[i+1])
            cent = []
            for j, npartons in enumerate(num_partons):
                if npartons > mul[i+1] and npartons < mul[i]:
                    p4x4 = f['event%s'%j]
                    impb = p4x4.attrs['impact_parameter']
                    nwnd = p4x4.attrs['num_of_participants']
                    cent.append([j, impb, nwnd, npartons])
            cent_key = 'cent/%s_%s'%(int(bins[i]), int(bins[i+1]))
            #del f[cent_key]
            #event_ids = f.create_dataset(cent_key, (len(cent), 4))
            event_ids = f[cent_key]
            event_ids[...] = np.array(cent)
            event_ids.attrs['mul_high'] = mul[i]
            event_ids.attrs['mul_low'] = mul[i+1]


#renew_num_partons()
#renew_centrality()
#renew_centrality_1()

for i in range(50):
    read_p4x4(idx=i)


