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

def write_config(configs, comments=''):
    '''write the current setting to hydro.info in the output directory'''
    fPathOut = configs.fPathOut
    if not os.path.exists(fPathOut):
        os.makedirs(fPathOut)

    configfile_name = os.path.join(fPathOut, 'hydro.info')
    #if not os.path.isfile(configfile_name):
    with open(configfile_name, 'w') as cfgfile:
        # Create the configuration file as it doesn't exist yet
        # Add content to the file
        Config = configparser.ConfigParser()
        Config.add_section('path')
        Config.set('path', 'fPathIni', configs.fPathIni)
        Config.set('path', 'fPathOut', configs.fPathOut)
        Config.add_section('glauber')
        Config.set('glauber', 'Edmax', configs.Edmax)
        Config.set('glauber', 'A', configs.A)
        Config.set('glauber', 'SQRTS', configs.SQRTS)
        Config.set('glauber', 'NucleonDensity', configs.NucleonDensity)
        Config.set('glauber', 'Ra', configs.Ra)
        Config.set('glauber', 'Eta', configs.Eta)
        Config.set('glauber', 'Si0', configs.Si0)
        Config.set('glauber', 'ImpactParameter', configs.ImpactParameter)
        Config.set('glauber', 'Hwn', configs.Hwn)
        Config.set('glauber', 'Eta_flat', configs.Eta_flat)
        Config.set('glauber', 'Eta_gw', configs.Eta_gw)

        Config.add_section('geometry')
        Config.set('geometry', 'NX', configs.NX)
        Config.set('geometry', 'NY', configs.NY)
        Config.set('geometry', 'NZ', configs.NZ)
        Config.set('geometry', 'ntskip', configs.ntskip)
        Config.set('geometry', 'nxskip', configs.nxskip)
        Config.set('geometry', 'nyskip', configs.nyskip)
        Config.set('geometry', 'nzskip', configs.nzskip)
        Config.set('geometry', 'DT', configs.DT)
        Config.set('geometry', 'DX', configs.DX)
        Config.set('geometry', 'DY', configs.DY)
        Config.set('geometry', 'DZ', configs.DZ)

        Config.add_section('intrinsic')
        Config.set('intrinsic', 'TAU0', configs.TAU0)
        Config.set('intrinsic', 'IEOS', configs.IEOS)
        Config.set('intrinsic', 'TFRZ', configs.TFRZ)
        Config.set('intrinsic', 'ETAOS', configs.ETAOS)
        Config.set('intrinsic', 'LAM1', configs.LAM1)
        Config.set('intrinsic', 'BSZ', configs.BSZ)

        Config.write(cfgfile)
        cfgfile.write('#comments: '+comments)


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

    parser.add_argument('--gubser_visc_test', nargs='?', const=1, type=bool, 
            default=False, help='true to switch to 2nd order gubser visc test')

    parser.add_argument('--use_float32', nargs='?', const=1, type=bool, 
            default=True, help='true for float and false for double precision')

    parser.add_argument('--save_to_hdf5', nargs='?', const=1, type=bool, 
            default=True, help='true to save bulkinfo to hdf5 file, otherwise save to .txt file')

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
