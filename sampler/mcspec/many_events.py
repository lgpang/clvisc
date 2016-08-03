#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fr 29 Apr 2016 11:44:06 CEST

from sampler import main
import time

if __name__=='__main__':
    t1 = time.time()
    for eid in range(302, 312):
        fpath = "/lustre/nyx/hyihp/lpang/auau200_results/cent0_5/etas0p08/event%s/"%eid
        viscous_on = 'true'
        force_decay = 'true'
        main(fpath, viscous_on, force_decay)

    t2 = time.time()

    print('it takes ', t2 - t1, 's for 10 events')


