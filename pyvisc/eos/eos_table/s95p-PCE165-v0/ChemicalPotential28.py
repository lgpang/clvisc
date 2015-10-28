#!/usr/bin/python

def lint(x1, y1, x2, y2, x):
   """Do linear interpolation """
   w1 = (x-x1)/(x2-x1)
   w2 = (x2-x)/(x2-x1)
   return w1*y2 + w2*y1


def ChemFrz_28(efrz):
    """ return the array of Chemical potential for 28 stable particles 
        at the freeze out energy """
    fchemical = open("s95p-PCE165-v0_pichem1.dat")
    e0 = float(fchemical.readline())
    de,ne = fchemical.readline().split()
    de,ne = float(de), int(ne)
    
    nstable = fchemical.readline()
    #print "e0=",e0, "de=", de, "ne=", ne, "nstable=", nstable
    energy = []
    chem = []
    for i in range(ne):
        energy.append(e0 + (ne-i-1)*de)
        chem.append(fchemical.readline())

    i = 1
    while energy[i] > efrz:
        i = i + 1
    
    c28l = chem[i-1].split()
    c28h = chem[i].split()
    
    chfrz = [] 
    for n in range(28):
        chfrz.append(lint( energy[i-1], float(c28l[n]), energy[i], float(c28h[n]), efrz ))

    return chfrz

def GetStable():
    """Get the stable particles pid from particles.dat,
    Gamma is exculded,
    return the array of 28 stable particles pid"""
    particles = open("particles.dat","r").readlines()
    stables = []
    for particle in particles:
        info = particle.split()
        if info[-1] == "Stable":
            if info[1] != "Gamma":
                stables.append(info[1])
    return stables

def Calc_Resonances_Chem(efrz):
    """Calc Resonances chemical potential from stable particles 
    chemical potential and the decay chain listed in pdg05.dat """
    Stables = GetStable()
    Chem28  = ChemFrz_28(efrz)
    Chem = {}
    for i in range(len(Stables)):
        Chem[Stables[i]] = Chem28[i]
    Chem['22'] = 0.0
    #print Chem

    fout = open("ChemForReso.dat","w")

    lines = open("pdg05.dat","r").readlines()
    i = 0
    while i in range(len(lines)):
        particle = lines[i].split()
        pid = particle[0]
        #print pid
        ndecays = int(particle[-1])
        mu_reso = 0.0
        for j in range(ndecays):
            decay = lines[i+j+1].split()
            ndaughter = int(decay[1])
            branch_ratio = float(decay[2])
            for k in range(ndaughter):
                mu_reso = mu_reso + branch_ratio*Chem[decay[3+k]]
            # mu_i = sum_j mu_j * n_ij
        Chem[pid] = mu_reso
        i = i + ndecays + 1
        print >>fout, pid, Chem[pid]
    return Chem
        

if __name__ == "__main__":
    import sys
    hydro_setting = open("../Vishydro1.inp", "r").readlines()
    efrz = hydro_setting[4].split()[0]
    efrz = float(efrz)
    print efrz

    if len(sys.argv) == 2:
        efrz = float(sys.argv[1])
        #freeze out energy density used to calc frz out chemical potential
    #print ChemFrz_28(efrz)
    #print GetStable()
    Chem = Calc_Resonances_Chem(efrz)
        
