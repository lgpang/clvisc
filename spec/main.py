#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 24 Apr 2015 15:17:45 CEST

#import matplotlib.pyplot as plt
import numpy as np

import spec_new as spec

if __name__=='__main__':
    import sys
    if not len(sys.argv) == 2:
        print("usage: python main.py event_path")
        exit(0)
    else :
        event_path = sys.argv[1]

    ylo, yhi = -0.5, 0.5
    ep_window = (3.3, 4.8)

    #charged = spec.Spec(event_path, pid='Charged', rapidity_kind='Eta')
    #charged.get_vn(ylo=ylo, yhi=yhi, event_plane_window=ep_window)

    pion = spec.Spec(event_path, pid='211', reso=False, rapidity_kind='Eta') 
    pion.get_vn(ylo=ylo, yhi=yhi, event_plane_window=ep_window)

    kaon = spec.Spec(event_path, pid='321', reso=False, rapidity_kind='Eta')
    kaon.get_vn(ylo=ylo, yhi=yhi, event_plane_window=ep_window)

    proton = spec.Spec(event_path, pid='2212', reso=False, rapidity_kind='Eta')
    proton.get_vn(ylo=ylo, yhi=yhi, event_plane_window=ep_window)

    #D0 = spec.Spec(event_path, pid='99999', reso=False, rapidity_kind='Eta')
    #D0.get_vn(ylo=ylo, yhi=yhi, event_plane_window=ep_window)

    # Get (1/2pi)dNdYPtdPt
    ylo_ptspec, yhi_ptspec = -0.8, 0.8

    pion = spec.Spec(event_path, pid='211', reso=False, rapidity_kind='Y')
    pion.get_ptspec(ylo_ptspec, yhi_ptspec)

    kaon = spec.Spec(event_path, pid='321', reso=False, rapidity_kind='Y')
    kaon.get_ptspec(ylo_ptspec, yhi_ptspec)

    proton = spec.Spec(event_path, pid='2212', reso=False, rapidity_kind='Y')
    proton.get_ptspec(ylo_ptspec, yhi_ptspec)

