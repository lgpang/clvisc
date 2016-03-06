#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

'''make plots for the data stored in bulkinfo.h5
   1d plots for avg/*   bulk1d/* 
   2d plots for bulk2d/*
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import sys
from time import time
import matplotlib.pyplot as plt
import h5py
import argparse

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from common_plotting import smash_style

# cloured output for files in bulkinfo.h5

colours={"default":"",
         "blue":"\x1b[01;34m",
         "cyan":   "\x1b[01;36m",
         "green":  "\x1b[01;32m",
         "red":    "\x1b[01;05;37;41m"}

#following from Python cookbook, #475186
def has_colours(stream):
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except:
        # guess false in case of error
        return False

has_colours = has_colours(sys.stdout)

def color_ls(coord, avg, bulk1d, bulk2d):
    ''' print out all the names for the data in hdf5 file
        in different colors if has_colours==True'''
    if has_colours:
        # print the coordinates
        sys.stdout.write(colours['blue'] + 'coord/' + "\x1b[00m") 
        for idx, item in enumerate(coord):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(colours['green'] + space + item + "\x1b[00m") 
        # print the files for average or cent data as a function of time
        sys.stdout.write(colours['blue'] + '\navg/' + "\x1b[00m") 
        for idx, item in enumerate(avg):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(colours['green'] + space + item + "\x1b[00m") 
        # print the files for 1d data as a function of time
        sys.stdout.write(colours['blue'] + '\nbulk1d/' + "\x1b[00m") 
        for idx, item in enumerate(bulk1d):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(colours['green'] + space + item + "\x1b[00m") 
        # print the files for 2d data as a function of time
        sys.stdout.write(colours['blue'] + '\nbulk2d/' + "\x1b[00m") 
        for idx, item in enumerate(bulk2d):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(colours['green'] + space + item + "\x1b[00m") 
        sys.stdout.write('\n') 
    else:
        # print the coordinates
        sys.stdout.write('coord/')
        for idx, item in enumerate(coord):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(space + item)
        # print the files for average or cent data as a function of time
        sys.stdout.write('\navg/')
        for idx, item in enumerate(avg):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(space + item)
        # print the files for 1d data as a function of time
        sys.stdout.write('\nbulk1d') 
        for idx, item in enumerate(bulk1d):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(space + item)
        sys.stdout.write('\nbulk2d') 
        for idx, item in enumerate(bulk2d):
            space = '\n' if not idx%5 else '\t\t'
            sys.stdout.write(space + item)
        sys.stdout.write('\n') 


# get all data entries in hdf5 file
def get_all_files(f_h5):
    '''list all data names that can be plotted'''
    coord, avg, bulk1d, bulk2d = [], [], [], []
    for name in f_h5['coord']:
        coord.append(name)

    for name in f_h5['avg']:
        avg.append(name)

    for name in f_h5['bulk1d']:
        bulk1d.append(name)

    for name in f_h5['bulk2d']:
        bulk2d.append(name)

    return coord, avg, bulk1d, bulk2d


def plot(fname, coord, avg, bulk1d, bulk2d,
         save_fig = False, save_data = False):
    '''make 1d plot for data in bulk1d/ and avg/
       make contour plot for data in bulk2d/'''
    x = f_h5['coord/x'][...]
    y = f_h5['coord/y'][...]
    z = f_h5['coord/etas'][...]
    tau = f_h5['coord/tau'][...]
    xy_ext = (x[0], x[-1], y[0], y[-1])
    xz_ext = (x[0], x[-1], z[0], z[-1])
    yz_ext = (y[0], y[-1], z[0], z[-1])

    fig_name = './' + fname + '.pdf'

    if fname is not '':
        if fname in avg:
            fname = 'avg/%s'%fname
            data = f_h5[fname][...]
            xlabel = r'$\tau\ [fm]$'
            ylabels = [r'$T(0,0,0)\ [GeV]$', r'$eccp$', r'$ed(0,0,0)\ [GeV/fm^3]$',
                       r'$dS/dY\ [fm^{-3}]$', r'$<v_r>$']
            x_coord = tau
            kind = ['Tcent', 'eccp', 'edcent', 'entropy', 'vr']
            for idx, substr in enumerate(kind):
                if substr in fname:
                    plt.plot(x_coord, data)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabels[idx])
                    plt.subplots_adjust(bottom=0.15)

        if fname in bulk1d:
            fname = 'bulk1d/%s'%fname
            data = f_h5[fname][...]
            xlabels = [r'$\eta_s$', r'$\eta_s$', r'$x\ [fm]$',
                       r'$y\ [fm]$', r'$\eta_s$', r'$x\ [fm]$',
                       r'$y\ [fm]$', r'$\eta_s$']

            ylabels = [r'$v_1$', r'$eccp$',
                       r'$ed(x,y=0,\eta_s=0)\ [GeV/fm^3]$',
                       r'$ed(x=0,y,\eta_s=0)\ [GeV/fm^3]$',
                       r'$ed(x=0,y=0,\eta_s)\ [GeV/fm^3]$',
                       r'$v_x(x,y=0,\eta_s=0)$',
                       r'$v_y(x=0,y,\eta_s=0)$',
                       r'$\tau \times v_{\eta}(x=0,y=0,\eta_s)$']

            pos = fname.find('tau')
            time_stamp = fname[pos+3:].replace('p', '.')
            x_coord = [z, z, x, y, z, x, y, z]

            kind = ['eccp1', 'eccp2', 'ex', 'ey', 'ez',
                    'vx', 'vy', 'vz']

            for idx, substr in enumerate(kind):
                if substr in fname:
                    plt.plot(x_coord[idx], data)
                    plt.xlabel(xlabels[idx])
                    plt.ylabel(ylabels[idx])
                    plt.title(r'$\tau=%s\ [fm]$'%time_stamp)
                    plt.subplots_adjust(bottom=0.15)

        if fname in bulk2d:
            fname = 'bulk2d/%s'%fname
            data = f_h5[fname][...]
            pos = fname.find('tau')
            time_stamp = fname[pos+3:].replace('p', '.')

            xlabels = [r'$x\ [fm]$', r'$x\ [fm]$', r'$y\ [fm]$',
                       r'$x\ [fm]$', r'$x\ [fm]$', r'$y\ [fm]$',
                       r'$x\ [fm]$', r'$x\ [fm]$', r'$y\ [fm]$',
                       r'$x\ [fm]$', r'$x\ [fm]$', r'$y\ [fm]$']

            ylabels = [r'$y\ [fm]$', r'$\eta_s$', r'$\eta_s$',
                       r'$y\ [fm]$', r'$\eta_s$', r'$\eta_s$',
                       r'$y\ [fm]$', r'$\eta_s$', r'$\eta_s$',
                       r'$y\ [fm]$', r'$\eta_s$', r'$\eta_s$']

            titles = [ r'$ed\ [GeV/fm^3]\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$ed\ [GeV/fm^3]\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$ed\ [GeV/fm^3]\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_x\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_x\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_x\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_y\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_y\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$v_y\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$\tau\times v_{\eta}\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$\tau\times v_{\eta}\ @\ \tau=%s\ fm$'%time_stamp,
                       r'$\tau\times v_{\eta}\ @\ \tau=%s\ fm$'%time_stamp]

            kind = ['exy', 'exz', 'eyz',
                    'vx_xy', 'vx_xz', 'vx_yz',
                    'vy_xy', 'vy_xz', 'vy_yz',
                    'vz_xy', 'vz_xz', 'vz_yz']

            extents = [xy_ext, xz_ext, yz_ext, 
                       xy_ext, xz_ext, yz_ext, 
                       xy_ext, xz_ext, yz_ext, 
                       xy_ext, xz_ext, yz_ext]

            params = {'figure.figsize': (15., 12.)}
            plt.rcParams.update(params)

            for idx, substr in enumerate(kind):
                if substr in fname:
                    plt.imshow(data.T, origin='lower', extent=extents[idx],
                            aspect='auto')

                    plt.xlabel(xlabels[idx])
                    plt.ylabel(ylabels[idx])
                    plt.title(titles[idx])
                    plt.colorbar()
                    plt.subplots_adjust(bottom=0.15)

    smash_style.set()
    if save_fig:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=\
        '''make plots for bulkinfo, read from bulkinfo.h5
         Usage: python h5view.py your_directory_for_bulkinfo.py
                output: ls the files in bulkinfo.py
                python h5view.py your_path --plot data_name
                output: 1d or 2d plots for the bulk data''')
    
    parser.add_argument('--path', nargs='?', const=1,  default='',
            help='directory for the bulkinfo.h5')

    parser.add_argument('--save_fig', nargs='?', const=1, type=bool,
            default=False, help='true to save figure instead of show')

    parser.add_argument('--save_data', nargs='?', const=1, type=bool,
            default=False, help='true to save the data to txt file')


    args, unknown = parser.parse_known_args()

    # directory for the bulkinfo.h5 file
    fpath = '../results/event0'
    if args.path is not '':
        fpath = args.path

    f_h5 = h5py.File(fpath+'/bulkinfo.h5', 'r')

    data_name = 'exz_tau1p5'

    if len(sys.argv) == 2:
        data_name = sys.argv[1]

    coord, avg, bulk1d, bulk2d = get_all_files(f_h5)
    color_ls(coord, avg, bulk1d, bulk2d)

    plot(data_name, coord, avg, bulk1d, bulk2d, args.save_fig, args.save_data)

    if args.save_data:
        save_txt(data_name)

    f_h5.close()
