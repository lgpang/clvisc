#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob

import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc
from glauber import Glauber, weight_mean_b

import matplotlib.pyplot as plt
from numba import jit
import math

np.random.seed(0)

num_events = 1000

# get random angles for the jet direction 
random_angles = np.random.rand(num_events) * 2 * np.pi

np.savetxt('random_angles.txt', random_angles)
print(random_angles[:10])
#plt.hist(random_angles, bins=50)
#plt.show()

# get jet start position randomly
def random_jet_position_index(NX, NY, NZ, DX, DY, num_events):
    nbc = np.loadtxt('run2/cent_20_30_etaos0.0/nbin.dat').flatten()
    prob = nbc / nbc.sum()
    jet_pos_index = np.random.choice(prob.size, size=num_events, p=prob)
    def from_nbin_coord_to_lattice_index(nbin_index):
        ix = nbin_index // 66
        iy = nbin_index - 66 * ix
        xx = -9.75 + ix * 0.3 
        yy = -9.75 + iy * 0.3 
        x0 = - NX // 2 * DX
        y0 = - NY // 2 * DY
        lx = ((xx - x0) / DX).astype(np.int32)
        ly = ((yy - y0) / DY).astype(np.int32)
        return lx * NY * NZ + ly * NZ + NZ // 2
    return from_nbin_coord_to_lattice_index(jet_pos_index)

random_pos_index = random_jet_position_index(301, 301, 121, DX=0.08, DY=0.08, num_events=num_events)

print(random_pos_index[:10])
scat_x = (random_pos_index) // (301 * 121)
scat_y = (random_pos_index - scat_x * 301 * 121 ) // 121
#plt.scatter(scat_x, scat_y)
#plt.xlim(0, 301)
#plt.ylim(0, 301)
#plt.axes().set_aspect('equal', 'datalim')
#plt.show()

def one_shot(fout, impact_parameter=7.8, etaos=0.0, with_eloss=False, gpuid=2, eventid=0):
    if not os.path.exists(fout):
        os.makedirs(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 121

    cfg.DT = 0.005
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DZ = 0.15
    cfg.IEOS = 1
    cfg.ntskip = 60
    cfg.nxskip = 4
    cfg.nyskip = 4
    cfg.nzskip = 2
    cfg.ImpactParameter = impact_parameter

    cfg.Hwn = 0.95

    cfg.TAU0 = 0.4
    cfg.Edmax = 55.0
    cfg.ETAOS_YMIN = etaos
    cfg.fPathOut = fout
    write_config(cfg)

    t0 = time()
    visc = CLVisc(cfg, gpu_id=gpuid)
    ini = Glauber(cfg, visc.ctx, visc.queue, visc.compile_options,
                  visc.ideal.d_ev[1], save_nbc=True)

    if with_eloss:
        jet_eloss_src = {'switch_on':True, 'start_pos_index':random_pos_index[eventid],
                         'direction':random_angles[eventid]}
    else:
        jet_eloss_src = {'switch_on':False, 'start_pos_index':random_pos_index[eventid],
                         'direction':random_angles[eventid]}

    visc.evolve(max_loops=2500, save_hypersf=True, save_bulk=True,
                jet_eloss_src=jet_eloss_src, force_run_to_maxloop=True)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

    cwd = os.getcwd()
    os.chdir('../CLSmoothSpec/build')
    os.system('cmake -D VISCOUS_ON=ON ..')
    os.system('make')
    call(['./spec', fout])
    os.chdir(cwd)
    after_reso = '0'
    call(['python', '../spec/main.py', fout, after_reso])
    os.chdir(cwd)


def main(event_id=0, gpuid=2, etaos=0.16, with_eloss='true'):
    fpath_out = os.path.abspath('/lustre/nyx/hyihp/lpang/jet_eloss')
    cmin, cmax = 20, 30
    b = weight_mean_b(cmin, cmax)
    if not os.path.exists(fpath_out):
        os.makedirs(fpath_out)

    if with_eloss == 'true':
        fout = os.path.join(fpath_out, 'cent_%s_%s_etaos%s_with/event%s'%(cmin, cmax, etaos, event_id))
        one_shot(fout, b, etaos, with_eloss=True, gpuid=gpuid, eventid=event_id)
    else:
        fout = os.path.join(fpath_out, 'cent_%s_%s_etaos%s_nojet/event%s'%(cmin, cmax, etaos, event_id))
        one_shot(fout, b, etaos, with_eloss=False, gpuid=gpuid, eventid=event_id)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 6:
        gpuid = int(sys.argv[1])
        etaos = float(sys.argv[2])
        with_eloss = sys.argv[3]
        start_eid = int(sys.argv[4])
        end_eid = int(sys.argv[5])
        for eid in range(start_eid, end_eid):
            main(eid, gpuid, etaos, with_eloss)


