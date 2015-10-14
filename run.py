#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Wed 14 Oct 2015 11:22:11 CEST

import numpy as np
from subprocess import call
import os

call(['python', 'pyvisc/ideal.py'])
cwd = os.getcwd()
os.chdir('CLSmoothSpec/build')
call(['./spec', '../../results/event0', '0.137'])
os.chdir(cwd)
call(['python', 'spec/main.py', 'results/event0'])


