#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 07 Jul 2016 02:13:05 AM CEST

import matplotlib.pyplot as plt
import numpy as np
from common_plotting import smash_style
import argparse
from vn_vs_eta_data import PbPb2760
import os
import pandas as pd


def cmp_ideal_vs_visc(dir_ideal='/lustre/nyx/hyihp/lpang/trento_ebe_hydro/pbpb2p76_results_ampt/etas0p0/', 
                      dir_visc='/lustre/nyx/hyihp/lpang/trento_ebe_hydro/pbpb2p76_results_ampt/etas0p16/',
                      centrality='0-5', num_events=200, subfig_label='A'):
    vn_ideal = np.empty((num_events, 20, 7))
    vn_visc = np.empty((num_events, 20, 7))
    vn_ideal[:] = np.NAN
    vn_visc[:] = np.NAN

    for eid in range(0, num_events):
        try:
            path_ideal = os.path.join(dir_ideal, centrality.replace('-','_'), 'event%d'%eid, 'vn24_vs_eta.txt')
            vn_ideal[eid] = np.loadtxt(path_ideal)
            path_visc = os.path.join(dir_visc, centrality.replace('-','_'), 'event%d'%eid, 'vn24_vs_eta.txt')
            vn_visc[eid] = np.loadtxt(path_visc)
        except BaseException as e:
            print("%s when load data for event %s"%(str(e), eid))

    # due to statistics, sometimes vn{2} or vn{4} has nan values, skip those events
    vn_ideal_mean = np.nanmean(vn_ideal, axis=0)
    vn_visc_mean = np.nanmean(vn_visc, axis=0)

    pbpb = PbPb2760()
    exp_v22 = pbpb.get(centrality, 'v22')
    exp_v32 = pbpb.get(centrality, 'v32')

    ratio = vn_visc_mean[:, 1].sum() / vn_ideal_mean[:, 1].sum() 
    plt.plot(vn_visc_mean[:, 0], vn_visc_mean[:, 1], 'r-', label=r'$CLVisc\ \eta/s=0.16$')
    plt.plot(vn_visc_mean[:, 0], vn_visc_mean[:, 2], 'b-')
    plt.plot(vn_ideal_mean[:, 0], ratio * vn_ideal_mean[:, 1], 'r--', label=r'$(CLVisc\ \eta/s=0)\times%.2f$'%ratio)
    plt.plot(vn_ideal_mean[:, 0], ratio * vn_ideal_mean[:, 2], 'b--')

    plt.errorbar(exp_v22[0], exp_v22[1], yerr=(exp_v22[4], exp_v22[5]), elinewidth=20, ls='', color='r', label='ALICE')
    plt.errorbar(exp_v32[0], exp_v32[1], yerr=(exp_v32[4], exp_v32[5]), elinewidth=20, ls='', color='b')

    plt.text(5.5, 1.0*vn_visc_mean[-1, 1], r'$v_2\{2\}$', color='r')
    plt.text(5.5, 1.0*vn_visc_mean[-1, 2], r'$v_3\{2\}$', color='b')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$v_n\{2\}$')
    #plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV,\ 20-30\%$')
    plt.xlim(-6.0, 7.0)
    plt.ylim(0.0, 1.5*vn_visc_mean[:, 1].max())
    plt.text(0.7, 0.9, '('+subfig_label+') '+centrality+r'$\%$', transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    smash_style.set(line_styles=False)

    plt.tight_layout()

    #plt.legend(ncol=2, loc="upper left", bbox_to_anchor=[0, 1])
    plt.legend(loc="upper left")
    plt.savefig('figs/vn2_vs_eta_{cent}.pdf'.format(cent=centrality.replace('-', '_')))
    plt.show()

    # save mean vn(eta) from hydro calculation to data file
    hydro_data_save_path = os.path.join(os.getcwd(), 'figs/hydro_pbpb2760_vn_vs_eta/')
    if not os.path.exists(hydro_data_save_path): os.makedirs(hydro_data_save_path)
    df = pd.DataFrame({"eta" : vn_ideal_mean[:, 0],
                       "v22" : vn_ideal_mean[:, 1],
                       "v32" : vn_ideal_mean[:, 2],
                       "v42" : vn_ideal_mean[:, 3]})
    df.to_csv(os.path.join(hydro_data_save_path, 'etaos0p0_cent%s.csv'%centrality.replace('-', '_')))
    df2 = pd.DataFrame({"eta" : vn_visc_mean[:, 0],
                       "v22" : vn_visc_mean[:, 1],
                       "v32" : vn_visc_mean[:, 2],
                       "v42" : vn_visc_mean[:, 3]})
    df2.to_csv(os.path.join(hydro_data_save_path, 'etaos0p16_cent%s.csv'%centrality.replace('-', '_')))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_events", type=int, default=200, help="num of events that will use")
    parser.add_argument("--input_dir", help="path to folder containing event*/")
    parser.add_argument("--etaos", default='0p16', help="eta/s string 0p0 or 0p16 in the name of the output figure")
    parser.add_argument("--centrality", choices=['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60'], help="centrality range")
    args = parser.parse_args()

    print(args.num_events)

    vn = np.empty((args.num_events, 20, 7))
    vn[:] = np.NAN

    for eid in range(0, args.num_events):
        try:
            vn_new = np.loadtxt('%s/event%d/vn24_vs_eta.txt'%(args.input_dir, eid))
            vn[eid] = vn_new
        except:
            print("no data for event %s"%eid)

    vn_mean = np.nanmean(vn, axis=0)

    pbpb = PbPb2760()
    exp_v22 = pbpb.get(args.centrality, 'v22')
    exp_v32 = pbpb.get(args.centrality, 'v32')

    plt.plot(vn_mean[:, 0], vn_mean[:, 1], 'r-', label='CLVisc')
    plt.plot(vn_mean[:, 0], vn_mean[:, 2], 'b-')
    #plt.plot(vn_mean[:, 0], vn_mean[:, 3], 'g:', label=r'CLVisc, n=4, $\eta/s=0.08$')

    plt.errorbar(exp_v22[0], exp_v22[1], yerr=(exp_v22[4], exp_v22[5]), elinewidth=20, ls='', color='r', label='ALICE')
    plt.errorbar(exp_v32[0], exp_v32[1], yerr=(exp_v32[4], exp_v32[5]), elinewidth=20, ls='', color='b')

    plt.text(5.5, 1.0*vn_mean[-1, 1], r'$v_2\{2\}$', color='r')
    plt.text(5.5, 1.0*vn_mean[-1, 2], r'$v_3\{2\}$', color='b')

    plt.xlabel(r'$\eta$')
    plt.ylabel(r'$v_n\{2\}$')
    #plt.title(r'$Pb+Pb\ \sqrt{s_{NN}}=2.76\ TeV,\ 20-30\%$')
    plt.xlim(-6.0, 7.0)
    plt.text(0.4, 0.15, args.centrality+r'$\%$', transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    smash_style.set()

    plt.tight_layout()

    #plt.legend(ncol=2, loc="upper left", bbox_to_anchor=[0, 1])
    plt.legend(loc="best")
    plt.savefig('figs/vn2_vs_eta_{cent}_etaos{etaos}.pdf'.format(cent=args.centrality.replace('-', '_'), etaos=args.etaos))
    plt.show()

    # save mean vn(eta) from hydro calculation to data file
    hydro_data_save_path = os.path.join(os.getcwd(), 'figs/hydro_pbpb2760_vn_vs_eta/')
    if not os.path.exists(hydro_data_save_path): os.makedirs(hydro_data_save_path)
    df = pd.DataFrame({"eta" : vn_mean[:, 0],
                       "v22" : vn_mean[:, 1],
                       "v32" : vn_mean[:, 2],
                       "v42" : vn_mean[:, 3]})
    df.to_csv(os.path.join(hydro_data_save_path, 'etaos%s_cent%s'%(args.etaos, args.centrality.replace('-', '_'))))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_events", type=int, default=200, help="num of events that will use")
    parser.add_argument("--centrality", choices=['0-5', '5-10', '10-20', '20-30', '30-40', '40-50', '50-60'], help="centrality range")
    parser.add_argument("--subfig", default='', help="(A), (B) ... labels for subfigs for publication")
    args = parser.parse_args()

    cmp_ideal_vs_visc(centrality=args.centrality, num_events=args.num_events, subfig_label=args.subfig)
