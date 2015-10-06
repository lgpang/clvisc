#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

from subprocess import call

if __name__ == '__main__':
    for eventid in xrange(0, 100):
        call(['python', 'main.py', '/scratch/hyihp/pang/ini/AuAu_Ini_b0_5_sig0p6/event%d'%eventid])
        call(['python', 'main.py', '/scratch/hyihp/pang/ini/AuAu_Ini_b5_10_sig0p6/event%d'%eventid])
        call(['python', 'main.py', '/scratch/hyihp/pang/ini/AuAu_Ini_b10_20_sig0p6/event%d'%eventid])
        call(['python', 'main.py', '/scratch/hyihp/pang/ini/AuAu_Ini_b30_40_sig0p6/event%d'%eventid])
        print eventid, 'finished'
