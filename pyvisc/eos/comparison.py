import matplotlib.pyplot as plt
import wb
import glueball


def p_vs_e_around_Tc():
    plt.plot(wb.ed[:200], wb.pr[:200], 'r-')
    plt.plot(glueball.ed[:200], glueball.pr[:200], 'b--')
    plt.xlabel(r'energy density $[GeV/fm^3]$')
    plt.ylabel(r'pressure $[GeV/fm^3]$')
    plt.show()


def p_vs_e_above_Tc():
    plt.plot(wb.ed[200:], wb.pr[200:], 'r-')
    plt.plot(glueball.ed[200:], glueball.pr[200:], 'b--')
    plt.xlabel(r'energy density $[GeV/fm^3]$')
    plt.ylabel(r'pressure $[GeV/fm^3]$')
    plt.show()



if __name__ == '__main__':
    #p_vs_e_around_Tc()
    p_vs_e_above_Tc()
