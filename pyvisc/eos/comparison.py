import matplotlib.pyplot as plt
import wb
import glueball
import eos


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


def p_vs_e():
    import numpy as np
    wb = eos.Eos(2)
    pce = eos.Eos(1)
    ed = np.linspace(0.01, 300, 500)
    plt.plot(ed, wb.f_S(ed), label='wb')
    plt.plot(ed, pce.f_S(ed), label='pce')
    plt.xlabel(r'energy density $[GeV/fm^3]$', fontsize=25)
    #plt.ylabel(r'pressure $[GeV/fm^3]$', fontsize=25)
    plt.ylabel(r'temperature $[GeV]$', fontsize=25)
    #plt.ylabel(r'entropy density $[fm^{-3}]$', fontsize=25)
    plt.legend(loc='best')
    plt.show()

def S_at_RHIC_LHC():
    import numpy as np
    wb = eos.Eos(2)
    pce = eos.Eos(1)
    ed = np.linspace(0.0, 500, 500)
    plt.plot(ed, wb.f_S(ed), label='wb')
    plt.xlabel(r'energy density $[GeV/fm^3]$', fontsize=25)
    plt.ylabel(r'entropy density $[fm^{-3}]$', fontsize=25)

    plt.axvline(32.0, color='k', linestyle= '--')
    plt.axhline(wb.f_S(32.0), color='k', linestyle='--')

    plt.axvline(98.0, color='r', linestyle= '--')
    plt.axhline(wb.f_S(98.0), color='r', linestyle='--')

    plt.text(32.0, 0.9*wb.f_S(32.0), 'RHIC')
    plt.text(98.0, 0.9*wb.f_S(98.0), 'LHC')

    plt.title(r'$\tau_0$=0.6 fm, entropy ratio=%.2f, multiplicity ratio=1650/650=%.2f'%(
                wb.f_S(98.0)/wb.f_S(32.0), 1600.0/650.0))

    plt.legend(loc='best', frameon=False)
    plt.show()



if __name__ == '__main__':
    #p_vs_e_around_Tc()
    #p_vs_e()
    #p_vs_e_above_Tc()
    S_at_RHIC_LHC()
