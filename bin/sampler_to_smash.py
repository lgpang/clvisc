#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 07 Apr 2018 11:44:03 PM CEST

import numpy as np
from subprocess import call
import os


def sample_plus_smash(event_path, viscous_on="true", nsampling=200):
    cwd = os.getcwd()
    # replace the pdg05.dat with the smash particle data table
    call(['cp', 'smash_pdg.dat', os.path.join(event_path, 'pdg05.dat')])
    #os.chdir('../pyvisc/')
    # get particle spectra from MC sampling and turn off resonance decay
    #call(['python', 'spec.py', '--event_dir', event_path,
    #  '--viscous_on', viscous_on, "--reso_decay", "false",
    #  "--nsampling", '%s'%nsampling, '--mode', 'mc'])
    os.chdir('../sampler/build')
    force_decay = 'false'
    call(['./main', event_path, viscous_on, force_decay, '%s'%nsampling])

    data_path_for_afterburner = os.path.join(event_path, 'afterburner_smash')
    if not os.path.exists(data_path_for_afterburner):
        os.makedirs(data_path_for_afterburner)
    else:
        import shutil
        shutil.rmtree(data_path_for_afterburner)
    os.chdir(cwd)
    with open('config_afterburner.yaml', 'r') as ftemplate:
        template = ftemplate.read()
        smash_config = template.format(num_over_sampling=nsampling,
                hydro_event_path=event_path)
        cfg_path = os.path.join(event_path, 'smash_config.yaml')
        with open(cfg_path, 'w') as fsmash_config:
            fsmash_config.write(smash_config)
        os.chdir('/lustre/nyx/hyihp/lpang/after_burner/build/')
        call(['./smash', '-i', cfg_path, '-o', data_path_for_afterburner])

if __name__=='__main__':
    sample_plus_smash(event_path='/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/auau200/15_16/event100/', nsampling=10)

