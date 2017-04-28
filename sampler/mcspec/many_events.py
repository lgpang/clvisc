#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 29 Apr 2016 11:44:06 CEST

from sampler import main
import time

if __name__=='__main__':
    t1 = time.time()
    for eid in range(0, 200):
        #fpath = "/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p0/event%s/"%eid
        #fpath = "/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent20_30/etas0p16/event%s/"%eid
        fpath = "/lustre/nyx/hyihp/lpang/new_polarization/pbpb2p76_results/cent0_5/etas0p16/event%s/"%eid
        try:
            viscous_on = 'true'
            force_decay = 'true'
            main(fpath, viscous_on, force_decay, nsampling=100)
            print(eid, 'finished')
        except:
            print(eid, ' hydro not finished')

    t2 = time.time()

    print('it takes ', t2 - t1, 's for 200 events')


