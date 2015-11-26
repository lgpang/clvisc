#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 20 Nov 2015 12:04:35 CET
import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))

from ideal import CLIdeal
from config import cfg, write_config
import pyopencl as cl
from glauber import Glauber
from eos.eos import Eos

def get_edmax(tau0, IEOS, ed_ref=30.0, tau_ref=0.6):
    '''get edmax for different tau0 to keep the dN/dEta unchanged
       using the relationship tau0*S0 = tau_ref*S_ref
       where S0 is the entropy density at (x=y=eta=0)'''
    from scipy.optimize import fsolve
    eos = Eos(IEOS)
    entropy = lambda ed: eos.f_S(ed)
    fzero = lambda ed: entropy(ed_ref)*tau_ref - entropy(ed)*tau0
    return fsolve(fzero, ed_ref)[0]


def squeezing(tau0, sys='RHIC', tdec=0.1, eB0=0.1, sigx=1.3, sigy=2.6, path_out='../results/event0'):
    cfg.IEOS = 2
    cfg.TAUD = tdec
    cfg.EB0 = eB0
    cfg.SIGX = sigx
    cfg.SIGY = sigy
    cfg.fPathOut = path_out

    cfg.TAU0 = tau0

    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 61
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DT = 0.02
    cfg.ntskip = 15
    cfg.nxskip = 2
    cfg.nyskip = 2

    if sys == 'RHIC':
        ed_ref = 30.0
        tau_ref = 0.6
        cfg.Edmax = get_edmax(tau0, cfg.IEOS, ed_ref, tau_ref)
    elif sys == 'LHC':
        ed_ref = 55.0
        tau_ref = 0.6
        cfg.Edmax = get_edmax(tau0, cfg.IEOS, ed_ref, tau_ref)

    cfg.ImpactParameter = 2.4

    if not os.path.exists(cfg.fPathOut):
        os.mkdir(cfg.fPathOut)

    write_config(cfg)

    ideal = CLIdeal(cfg, gpu_id=2)
    
    ini = Glauber(cfg, ideal.ctx, ideal.queue, ideal.gpu_defines, ideal.d_ev[1])

    ideal.evolve(max_loops=2000, save_bulk=True, save_hypersf=True)

    cwd = os.getcwd()
    
    from subprocess import call
    os.chdir('../CLSmoothSpec/build')
    #os.system('cmake -D VISCOUS_ON=ON ..')
    #os.system('make')
    call(['./spec', '../'+path_out])
    os.chdir(cwd)
    call(['python', '../spec/main.py', path_out])
    



if __name__ == '__main__':
    squeezing(tau0=0.6, sys='RHIC', tdec=1.9, eB0=0.0, path_out='../results/squeezing_tau0p6_td1p9_eb0p00/')
