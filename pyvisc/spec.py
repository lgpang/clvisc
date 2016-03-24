#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os, sys

if len(sys.argv) != 2:
    print 'usage:python run.py directory'
    exit()

path = os.path.abspath(sys.argv[1])

#call(['python', 'pyvisc/visc.py'])
cwd = os.getcwd()
os.chdir('../CLSmoothSpec/build')
os.system('cmake -D VISCOUS_ON=ON ..')
os.system('make')
call(['./spec', path])
os.chdir(cwd)
call(['python', '../spec/main.py', path])
