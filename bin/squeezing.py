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


def squeezing(tau0, sys='RHIC', b=10.0, tdec=0.1, eB0=0.1, sigx=1.3, sigy=2.6, path_out='../results/event0', gpu_id=2):
    cfg.IEOS = 2
    cfg.TAUD = tdec
    cfg.EB0 = eB0
    cfg.SIGX = sigx
    cfg.SIGY = sigy
    cfg.fPathOut = path_out

    cfg.TAU0 = tau0

    cfg.NX = 401
    cfg.NY = 401
    cfg.NZ = 61
    cfg.DX = 0.08
    cfg.DY = 0.08
    cfg.DT = 0.01
    cfg.ntskip = 30
    cfg.nxskip = 4
    cfg.nyskip = 4
    cfg.ImpactParameter = b
    cfg.ETAOS = 0.08

    if sys == 'RHIC':
        cfg.A = 197
        cfg.Ra = 6.54
        cfg.SQRTS = 200.0
        cfg.Si0 = 4.0
        ed_ref = 55.0
        tau_ref = 0.4
        cfg.Eta_gw = 0.4
        cfg.Edmax = get_edmax(tau0, cfg.IEOS, ed_ref, tau_ref)
    elif sys == 'LHC':
        ed_ref = 98.0
        tau_ref = 0.6
        cfg.Eta_flat = 3.5
        cfg.Eta_gw = 0.6
        cfg.Edmax = get_edmax(tau0, cfg.IEOS, ed_ref, tau_ref)
        cfg.A = 208
        cfg.SQRTS = 2760
        cfg.Ra = 6.62
        cfg.Eta = 0.546
        cfg.Si0 = 6.4


    if not os.path.exists(cfg.fPathOut):
        os.mkdir(cfg.fPathOut)

    write_config(cfg)

    ideal = CLIdeal(cfg, gpu_id=gpu_id)
    
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
    


def lifetime_dependence():
    '''lifetime dependence'''
    squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=1.8, sigy=3.6, path_out='../results/td_dependence/squeezing_pbpb276_tau0p2_td1p9_eb0p33/')
    squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.1, eB0=1.33, sigx=1.8, sigy=3.6, path_out='../results/td_dependence/squeezing_pbpb276_tau0p2_td1p1_eb1p33/')
    squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=0.5, eB0=1.33, sigx=1.8, sigy=3.6, path_out='../results/td_dependence/squeezing_pbpb276_tau0p2_td0p5_eb1p33/')
    squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=0.1, eB0=1.33, sigx=1.8, sigy=3.6, path_out='../results/td_dependence/squeezing_pbpb276_tau0p2_td0p1_eb1p33/')

def tau0_dependence():
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=0.00, sigx=1.3, sigy=2.6,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb0')
    squeezing(tau0=0.2, sys='RHIC', b=10.0, tdec=1.9, eB0=0.0, sigx=1.3, sigy=2.6,
              path_out='../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0')
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=1.3, sigy=2.6,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33')
    #squeezing(tau0=0.6, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=1.3, sigy=2.6,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p6_td1p9_eb1p33')
    squeezing(tau0=0.2, sys='RHIC', b=10.0, tdec=1.9, eB0=0.09, sigx=1.3, sigy=2.6,
              path_out='../results/tau0_dependence/squeezing_auau200_tau0p2_td1p9_eb0p09')
    squeezing(tau0=0.6, sys='RHIC', b=10.0, tdec=1.9, eB0=0.09, sigx=1.3, sigy=2.6,
              path_out='../results/tau0_dependence/squeezing_auau200_tau0p6_td1p9_eb0p09')


def sigma_dependence():
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=1.8, sigy=3.6,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33_sigy3p6')
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=1.5, sigy=3.0,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33_sigy3p0')
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=2.4, sigy=4.8,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33_sigy4p8_chiTcorrect')
    #squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=1.9, eB0=1.33, sigx=3.6, sigy=7.2,
    #          path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td1p9_eb1p33_sigy7p2_chiTcorrect')
    squeezing(tau0=0.2, sys='LHC', b=10.0, tdec=0.5, eB0=1.33, sigx=3.6, sigy=7.2,
              path_out='../results/tau0_dependence/squeezing_pbpb276_tau0p2_td0p5_eb1p33_sigy7p2_chiTcorrect')



def dNdEta_tau0():
    '''dNdEta for different tau0 at RHIC and LHC energy'''
    #squeezing(tau0=0.6, sys='RHIC', b=2.4, tdec=1.9, eB0=0.0, path_out='../results/squeezing_auau200_tau0p6_td1p9_eb0p00/')
    #squeezing(tau0=0.4, sys='RHIC', b=2.4, tdec=1.9, eB0=0.0, path_out='../results/squeezing_auau200_tau0p4_td1p9_eb0p00/')
    #squeezing(tau0=0.2, sys='RHIC', b=2.4, tdec=1.9, eB0=0.0, path_out='../results/squeezing_auau200_tau0p2_td1p9_eb0p00/')
    squeezing(tau0=0.6, sys='LHC', b=2.65, tdec=1.9, eB0=0.0, path_out='../results/squeezing_pbpb276_tau0p6_td1p9_eb0p00/')
    squeezing(tau0=0.4, sys='LHC', b=2.65, tdec=1.9, eB0=0.0, path_out='../results/squeezing_pbpb276_tau0p4_td1p9_eb0p00/')
    squeezing(tau0=0.2, sys='LHC', b=2.65, tdec=1.9, eB0=0.0, path_out='../results/squeezing_pbpb276_tau0p2_td1p9_eb0p00/')


if __name__ == '__main__':
    #lifetime_dependence()
    #tau0_dependence()
    sigma_dependence()

