##Read default configeration from hydro.info
##Update it with input from command line options
import numpy as np
import ConfigParser
from pyopencl import array
import argparse
import os

def read_config():
    '''read configeration from file, then update the value 
    with command line input if there is any'''
    _parser = ConfigParser.ConfigParser()
    
    cwd, cwf = os.path.split(__file__)
    _parser.read(os.path.join(cwd, 'hydro.info'))
    
    config = {}
    
    config['fPathIni'] = (_parser.get('path', 'fPathIni'), 
            'The absolute path for initial conditions')

    config['fPathOut'] = (_parser.get('path', 'fPathOut'), 
            'The absolute path for output directory')
    
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

    config['BSZ'] = ( _parser.getint( 'geometry', 'BSZ'), 
            'Local memory size in one dimension')
    
    config['DT'] = (_parser.getfloat( 'geometry', 'dt'), 
            'time step for hydro evolution' )

    config['DX'] = (_parser.getfloat( 'geometry', 'dx'), 
            'x step for hydro evolution' )

    config['DY'] = (_parser.getfloat( 'geometry', 'dy'), 
            'y step for hydro evolution' )

    config['DZ'] = (_parser.getfloat( 'geometry', 'dz'),
            'z step for hydro evolution' )
    
    config['TAU0'] = (_parser.getfloat('intrinsic', 'tau0'),
            'time when hydro starts')

    config['IEOS']  = (_parser.getint('intrinsic', 'IEOS'), 
            'EOS selection, 0 for ideal gas, 1 for s95p-ce, 2 for s95p-pce')

    config['TFRZ'] = (_parser.getfloat('intrinsic', 'TFRZ'), 
            'Freeze out temperature, default=0.137')

    config['ETAOS']= (_parser.getfloat('intrinsic', 'ETAOS'), 
            'Shear viscosity over entropy density')

    parser = argparse.ArgumentParser(description=\
        'Input parameters for hydrodynamic simulations')
    
    for key, value in config.items():
        parser.add_argument('--{key}'.format(key=key), nargs='?', const=1, 
                type=type(value[0]), default=value[0], help=value[1] )

    parser.add_argument('--use_float32', nargs='?', const=1, type=bool, 
            default=True, help='true for float and false for double precision')
    
    args, unknown = parser.parse_known_args()

    if args.use_float32 == True :
        args.real = np.float32
        args.real4 = array.vec.float4
        args.sz_real = np.dtype('float32').itemsize   #==sizeof(float) in c
        args.sz_real4 = array.vec.float4.itemsize
    else :
        args.real = np.float64
        args.real4 = array.vec.double4
        args.sz_real = np.dtype('float64').itemsize   #==sizeof(double) in c
        args.sz_real4= array.vec.double4.itemsize

    return args


cfg = read_config()
