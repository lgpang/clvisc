#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os, sys

if len(sys.argv) != 2:
    print('usage:python spec.py directory')
    exit()

path = os.path.abspath(sys.argv[1])

#call(['python', 'pyvisc/visc.py'])
cwd, cwf = os.path.split(__file__)
os.chdir(cwd)
os.chdir('../CLSmoothSpec/build')
os.system('cmake -D VISCOUS_ON=ON ..')
os.system('make')
call(['./spec', path])
os.chdir(cwd)
after_reso = '0'
call(['python', '../spec/main.py', path, after_reso])
os.chdir(cwd)
