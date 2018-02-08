#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 08 Feb 2018 04:29:04 AM CET

import numpy as np
import re


def regularize(comment):
    res = comment.replace('#', '').replace('\n','').strip()
    # remove spaces around '='
    res = re.sub(r'\s+=\s+', r'=', res)
    return res.split()

def get_comments(fname):
    '''get comment lines in file=fname,
    return a dictionary of options'''
    options = dict()
    with open(fname, 'r') as fin:
        for line in fin.readlines():
            if '#' in line and '=' in line:
                for opt in regularize(line):
                    opt_name, opt_value = opt.split('=')
                    options[opt_name] = float(opt_value)
    return options


if __name__=='__main__':
    options = get_comments('dat/00.dat')
    print(options['b'])
    print(options['ixcm'])

