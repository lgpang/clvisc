#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 20 Oct 2017 04:40:15 PM CST

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import pyopencl as cl
from pyopencl import array
import pyopencl.array as cl_array
from time import time
import argparse
import os, sys
from visc import CLVisc
import numpy as np

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)

parser = argparse.ArgumentParser(description='Input parameters for hydrodynamic simulations')

#### system setups
parser.add_argument('--riemann_test', nargs='?', const=1, type=bool, default=False, help='true to switch on riemann test for expansion to vacuum problem')
parser.add_argument('--gubser_visc_test', nargs='?', const=1, type=bool, default=False, help='true to switch to 2nd order gubser visc test')
parser.add_argument('--pimn_omega_coupling', nargs='?', const=1, type=bool, default=False, help='true to switch on pi{mu nu} and vorticity coupling term')
parser.add_argument('--omega_omega_coupling', nargs='?', const=1, type=bool, default=False, help='true to switch on vorticity and vorticity coupling term')
parser.add_argument('--use_float32', nargs='?', const=1, type=bool, default=True, help='true for float and false for double precision')
parser.add_argument('--save_to_hdf5', nargs='?', const=1, type=bool, default=True, help='true to save bulkinfo to hdf5 file, otherwise save to .txt file')
parser.add_argument('--opencl_interactive', nargs='?', const=1, type=bool, default=False, help='true to choose device type and device id at run time')

#### choose MC models to produce initial conditions for heavy ion collisions
parser.add_argument('--initial_condition', default='Glauber', choices=['ReadFromFile', 'Glauber', 'AMPT', 'Trento'])
parser.add_argument('--fPathOut', default='../results/auau200_cent0_5', help='The absolute path for output directory')
#### if choice == 'ReadFromFile', set the path to the initial condition 
parser.add_argument('--fPathIni', help='The absolute path for initial conditions')
#### if choice == 'Glauber', set parameters for optical Glauber model
parser.add_argument('--Edmax', type=float, default=55.0, help='maximum energy density for most central collisions')
parser.add_argument('--NumOfNucleons', type=float, default=197.0,  help='Number of nucleons, A=197 for Au; A=208 for Pb')
parser.add_argument('--SQRTS', type=float, default=200,  help='Beam energy in units of GeV/n; e.g. Au+Au 200 GeV SQRTS=200; Pb+Pb 2760 GeV, SQRTS=2760')
parser.add_argument('--NucleonDensity', type=float, default=0.17, help='With which the woods-saxon integration = 197 for A=197')
parser.add_argument('--Ra', type=float, default=6.38, help='Radius of the nucleus, Ra=6.38 for Au and 6.62 for Pb')
parser.add_argument('--Eta', type=float, default=0.535, help='Woods-Saxon tail parameter for nucleus, 0.535 for Au and 0.546 for Pb')
parser.add_argument('--Si0', type=float, default=4.0, help='Cross section for A+A collisions; 4.0 fm^2 for Au+Au 200 GeV and 6.4 fm^2 for Pb+Pb 30TeV')
parser.add_argument('--CentralityType', default='ImpactParameter', choices=['CentralityRange', 'ImpactParameter'],
                    help='Choose centrality range or impact parameter to determine the centrality in optical Glauber model')
parser.add_argument('--ImpactParameter', type=float, default=2.4, help='average impact parameter for one centrality bin. 2.4 for 0-5 Au+Au collisions')
parser.add_argument('--CentralityRange', default='0_5', help='Using CentralityRange to determine centralities in optical Glauber model')
parser.add_argument('--Hwn', type=float, default=0.95, help='dNdY propto Hwn*Npart + (1-Nwn)*Nbinary')
parser.add_argument('--Eta_flat', type=float, default=2.95, help='The width of the plateau along etas at mid rapidity')
parser.add_argument('--Eta_gw', type=float, default=0.5, help='the gaussian fall off width at large etas where fabs(etas)>Eta_flat/2')

#### Grid sizes, hyper surface cube sizes for numerical simulatioons
parser.add_argument('--NX', type=int, default=301, help='Grid size along x direction; x range = [-NX/2, NX/2] * DX')
parser.add_argument('--NY', type=int, default=301, help='Grid size along y direction;')
parser.add_argument('--NZ', type=int, default=161, help='Grid size along longitudinal direction; Here Z stands for space-time rapidity \eta_s')
parser.add_argument('--DT', type=float, default=0.01, help='Time step')
parser.add_argument('--DX', type=float, default=0.12, help='x cell size')
parser.add_argument('--DY', type=float, default=0.12, help='y cell size')
parser.add_argument('--DZ', type=float, default=0.12, help='space-time rapidity cell size')
parser.add_argument('--ntskip', type=int, default=36, help='Do output every ntskip time steps')
parser.add_argument('--nxskip', type=int, default=3, help='Do output every nxskip cells along x')
parser.add_argument('--nyskip', type=int, default=3, help='Do output every nyskip cells along y')
parser.add_argument('--nzskip', type=int, default=3, help='Do output every nzskip cells along eta_s')
parser.add_argument('--TAU0', type=float, default=0.4, help='Initial equilibrium time when hydro starts')
parser.add_argument('--IEOS', default=1, choices=[0, 1, 2, 3, 5], help='''Equation of state, 0: ideal gas p=e/3; 1: s95p_PCE lattice QCD EoS 2: WB2014 lattice QCD EoS.
3: Pure SU3 glue EoS. 5: EOSQ = with first order phase transition''')
parser.add_argument('--TFRZ', type=float, default=0.137, help='Freeze out temperature')
#### parametrization for temperature dependent eta/s
parser.add_argument('--ETAOS_XMIN', type=float, default=0.18, help='temperature for minimum eta/s(T)')
parser.add_argument('--ETAOS_YMIN', type=float, default=0.0, help='minimum eta/s(T)')
parser.add_argument('--ETAOS_LEFT_SLOP', type=float, default=0.0, help='slop of eta/s(T) when T < ETAOS_XMIN')
parser.add_argument('--ETAOS_RIGHT_SLOP', type=float, default=0.0, help='slop of eta/s(T) when T > ETAOS_XMIN')
####  if gubser_visc_test == True, add pimn^2 term whose coeficient is LAM1
parser.add_argument('--LAM1', type=float, default=-10.0, help='coefficient for pimn^2 term if gubser_visc_test==True')
parser.add_argument('--BSZ', type=int, default=64, help='Local workgroup size in each  dimension')
parser.add_argument('--GPU_ID', type=int, default=0, help='Choose which gpu to use if there are multiple GPUs per node')

cfg, unknown = parser.parse_known_args()

cfg.sz_int = np.dtype('int32').itemsize   #==sizeof(int) in c
if cfg.use_float32 == True :
    cfg.real = np.float32
    cfg.real4 = array.vec.float4
    cfg.real8 = array.vec.float8
    cfg.sz_real = np.dtype('float32').itemsize   #==sizeof(float) in c
    cfg.sz_real4 = array.vec.float4.itemsize
    cfg.sz_real8 = array.vec.float8.itemsize
else :
    cfg.real = np.float64
    cfg.real4 = array.vec.double4
    cfg.real8 = array.vec.double8
    cfg.sz_real = np.dtype('float64').itemsize   #==sizeof(double) in c
    cfg.sz_real4= array.vec.double4.itemsize
    cfg.sz_real8= array.vec.double8.itemsize


visc = CLVisc(cfg, gpu_id=cfg.GPU_ID)

if cfg.initial_condition == 'Glauber':
    visc.optical_glauber_ini()

visc.evolve(max_loops=2000, save_hypersf=True, save_bulk=False,
        force_run_to_maxloop=False, save_vorticity=False)

t1 = time()
print('finished. Total time: {dtime}'.format(dtime = t1-t0), file=sys.stdout)

from subprocess import call

# get particle spectra from MC sampling and force decay
call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
  '--viscous_on', "false", "--reso_decay", "true", "--nsampling", "2000",
  '--mode', 'mc'])

 # calc the smooth particle spectra
call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
  '--viscous_on', "false", "--reso_decay", "false", 
  '--mode', 'smooth'])
