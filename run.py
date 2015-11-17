#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os

#call(['python', 'pyvisc/visc.py'])
cwd = os.getcwd()
os.chdir('CLSmoothSpec/build')
os.system('cmake -D VISCOUS_ON=OFF ..')
os.system('make')
call(['./spec', '../../results/event_etaos0p08'])
os.chdir(cwd)
call(['python', 'spec/main.py', 'results/event_etaos0p08'])


