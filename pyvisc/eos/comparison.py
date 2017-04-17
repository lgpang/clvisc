import matplotlib.pyplot as plt
import wb_mod as wb
import glueball
from pce import CE
from pce import PCE


def p_vs_e_around_Tc():
    plt.plot(wb.ed[:200], wb.pr[:200], 'rs')
    plt.plot(glueball.ed[:200], glueball.pr[:200], 'bo')
    plt.xlabel(r'energy density $[GeV/fm^3]$')
    plt.ylabel(r'pressure $[GeV/fm^3]$')
    plt.show()


def p_vs_e_above_Tc():
    plt.plot(wb.ed[200:], wb.pr[200:], 'r-')
    plt.plot(glueball.ed[200:], glueball.pr[200:], 'b--')
    plt.xlabel(r'energy density $[GeV/fm^3]$')
    plt.ylabel(r'pressure $[GeV/fm^3]$')
    plt.show()



def wb_vs_pce(kind='pr'):
    if kind == 'pr':
        plt.plot(wb.ed[:], wb.pr[:], 'rs', label='wb')
        ed = PCE.eps_dat[0][:, 0]
        pr = PCE.eps_dat[0][:, 1]
        plt.plot(ed, pr, label='pce')
        plt.ylabel(r'pressure $[GeV/fm^3]$')
    elif kind == 'T':
        plt.plot( wb.ed[:], wb.T[:],'rs', label='wb')
        ed = PCE.eps_dat[0][:, 0]
        T = PCE.T_dat[0][:, 0]
        plt.plot(ed, T, label='pce')
        plt.ylabel(r'temperature $[GeV]$')
    plt.xlabel(r'energy density $[GeV/fm^3]$')
    plt.legend(loc='best')
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.1)
    plt.show()



if __name__ == '__main__':
    #p_vs_e_around_Tc()
    #p_vs_e_above_Tc()
    #wb_vs_pce('T')

    from eos import Eos
    pce150 = Eos(1)
    ed = pce150.f_ed(0.150)
    pr = pce150.f_P(ed)
    print('ed=', ed, 'pr=', pr, 'at T=0.137 GeV')
