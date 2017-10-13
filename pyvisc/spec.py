#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--event_dir", required=True, help="path to folder containing clvisc output")
parser.add_argument("--viscous_on", default='true', help="with non-quilibrium contribution df in Cooper-Frye particlization")
parser.add_argument("--reso_decay", default='true', help="true to switch_on resonance decay")
parser.add_argument("--nsampling", default='2000', help="number of over-sampling from one hyper-surface")
parser.add_argument("--mode", default='smooth', help="options:[smooth, mc]")
parser.add_argument("--gpu_id", default='0', help="for smooth spectra, one can choose gpu for parallel running")

cfg = parser.parse_args()

print(cfg.event_dir)
event_dir = os.path.abspath(cfg.event_dir)

src_dir = os.path.dirname(os.path.realpath(__file__))

if cfg.mode == 'smooth':
    dir_smooth_spec = os.path.join(src_dir, '../CLSmoothSpec/build')
    if not os.path.exists(dir_smooth_spec):
        os.makedirs(dir_smooth_spec)
    os.chdir(dir_smooth_spec)

    #os.system('rm -r *')
    #os.system('cmake ..')
    #os.system('make')
    call(['./spec', event_dir, cfg.viscous_on, cfg.reso_decay, cfg.gpu_id])
    os.chdir(src_dir)
    if cfg.reso_decay == 'true':
        call(['python', '../spec/main.py', event_dir, '1'])
    else:
        call(['python', '../spec/main.py', event_dir, '0'])
elif cfg.mode == 'mc':
    dir_mc_spec = os.path.join(src_dir, '../sampler/build')
    if not os.path.exists(dir_mc_spec):
        os.makedirs(dir_mc_spec)
    os.chdir(src_dir)
    dir_mc_spec = os.path.join(src_dir, '../sampler/mcspec/')
    os.chdir(dir_mc_spec)

    os.system('python sampler.py {path} {vis_on} {decay_on} {nsample}'.format(path=event_dir, 
                      vis_on=cfg.viscous_on, decay_on=cfg.reso_decay, nsample=cfg.nsampling))
    os.chdir(src_dir)
else:
    print("Choose from 'smooth' and 'mc' for spectra calc")
