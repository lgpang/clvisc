##Read default configeration from hydro.info
##Update it with input from command line options
import numpy as np
from pyopencl import array
import argparse
import os, sys

if sys.version_info <= (3, 0):
    import ConfigParser as configparser
else:
    import configparser

def read_config():
    '''read configeration from file, then update the value 
    with command line input if there is any'''
    _parser = configparser.ConfigParser()
    
    cwd, cwf = os.path.split(__file__)
    _parser.read(os.path.join(cwd, 'hydro.info'))
    
    config = {}
    
    # working directory
    config['fPathIni'] = (_parser.get('path', 'fPathIni'), 
            'The absolute path for initial conditions')

    config['fPathOut'] = (_parser.get('path', 'fPathOut'), 
            'The absolute path for output directory')

    # parameters for glauber initial conditions
    config['Edmax'] = (_parser.getfloat( 'glauber', 'Edmax'),
            'maximum energy density for most central collisions')

    config['A'] = (_parser.getfloat( 'glauber', 'A'),
            'Number of nucleons, A=197 for Au; A=208 for Pb')

    config['SQRTS'] = (_parser.getfloat( 'glauber', 'SQRTS'),
            'Beam energy in units of GeV/n; like Au+Au 200 GeV; Pb+Pb 2760 GeV, SQRTS=2760')

    config['NucleonDensity'] = (_parser.getfloat( 'glauber', 'NucleonDensity'),
            'With which the woods-saxon integration = 197 for A=197')

    config['Ra'] = (_parser.getfloat( 'glauber', 'Ra'),
            'Radius of the nucleus')

    config['Eta'] = (_parser.getfloat( 'glauber', 'Eta'),
            'woods-saxon diffusiveness parameter')

    config['Si0'] = (_parser.getfloat( 'glauber', 'Si0'),
            'inelastic scattering cross section')

    config['ImpactParameter'] = (_parser.getfloat( 'glauber', 'ImpactParameter'),
            'average impact parameter')

    config['Hwn'] = (_parser.getfloat( 'glauber', 'Hwn'),
            'in range [0,1), energy density contribution from number of wounded nucleons')

    config['Eta_flat'] = (_parser.getfloat( 'glauber', 'Eta_flat'),
            'The width of the plateau along etas at mid rapidity')

    config['Eta_gw'] = (_parser.getfloat( 'glauber', 'Eta_gw'),
            'the gaussian fall off at large etas where fabs(etas)>Eta_flat/2')
    
    # Grid sizes, hyper surface grain
    config['NX'] = (_parser.getint( 'geometry', 'NX'),
            'Grid size along x direction')

    config['NY'] = (_parser.getint( 'geometry', 'NY'),
            'Grid size along y direction')

    config['NZ'] = (_parser.getint( 'geometry', 'NZ'),
            'Grid size along z direction')
    
    config['ntskip'] = ( _parser.getint( 'geometry', 'ntskip'), 
            'Skip time steps for bulk information output'   )

    config['nxskip'] = ( _parser.getint( 'geometry', 'nxskip'), 
            'Skip steps along x for bulk information output')

    config['nyskip'] = ( _parser.getint( 'geometry', 'nyskip'), 
            'Skip steps along y for bulk information output')

    config['nzskip'] = ( _parser.getint( 'geometry', 'nzskip'), 
            'Skip steps along z for bulk information output')

   
    config['DT'] = (_parser.getfloat( 'geometry', 'DT'),
            'time step for hydro evolution' )

    config['DX'] = (_parser.getfloat( 'geometry', 'DX'),
            'x step for hydro evolution' )

    config['DY'] = (_parser.getfloat( 'geometry', 'DY'),
            'y step for hydro evolution' )

    config['DZ'] = (_parser.getfloat( 'geometry', 'DZ'),
            'z step for hydro evolution' )
    
    config['TAU0'] = (_parser.getfloat('intrinsic', 'TAU0'),
            'time when hydro starts')

    config['IEOS']  = (_parser.getint('intrinsic', 'IEOS'), 
            'EOS selection, 0 for ideal gas, 1 for s95p-pce,\
             2 for wuppertal budapest 2014 ce, 3 for glueball eos')

    config['TFRZ'] = (_parser.getfloat('intrinsic', 'TFRZ'), 
            'Freeze out temperature, default=0.137')

    config['ETAOS']= (_parser.getfloat('intrinsic', 'ETAOS'), 
            'Shear viscosity over entropy density')

    config['LAM1']= (_parser.getfloat('intrinsic', 'LAM1'), 
            'coefficient for pimn^2 term')

    config['BSZ'] = (_parser.getint('opencl', 'local_workgroup_size'), 
            'Local workgroup size in one dimension')

    parser = argparse.ArgumentParser(description=\
        'Input parameters for hydrodynamic simulations')
    
    for key, value in list(config.items()):
        parser.add_argument('--{key}'.format(key=key), nargs='?', const=1, 
                type=type(value[0]), default=value[0], help=value[1] )

    parser.add_argument('--use_float32', nargs='?', const=1, type=bool, 
            default=True, help='true for float and false for double precision')

    parser.add_argument('--opencl_interactive', nargs='?', const=1, type=bool, 
            default=False, help='true to choose device type and device id at run time')
    
    args, unknown = parser.parse_known_args()

    args.sz_int = np.dtype('int32').itemsize   #==sizeof(int) in c
    if args.use_float32 == True :
        args.real = np.float32
        args.real4 = array.vec.float4
        args.real8 = array.vec.float8
        args.sz_real = np.dtype('float32').itemsize   #==sizeof(float) in c
        args.sz_real4 = array.vec.float4.itemsize
        args.sz_real8 = array.vec.float8.itemsize
    else :
        args.real = np.float64
        args.real4 = array.vec.double4
        args.real8 = array.vec.double8
        args.sz_real = np.dtype('float64').itemsize   #==sizeof(double) in c
        args.sz_real4= array.vec.double4.itemsize
        args.sz_real8= array.vec.double8.itemsize

    return args


cfg = read_config()
