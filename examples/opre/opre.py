#!/usr/bin/env python
import sys
sys.path.append("C:\\Users\\adity\\Downloads\\USI\\Sem 3\\ECA\\pymlmc")
from pymlmc import mlmc_test, mlmc_plot
import matplotlib.pyplot as plt
import numpy
import numpy.random
from math import sqrt

class CallType(object):
    def __init__(self, name, M, N, L, Eps):
        self.name = name
        self.M = M # refinement cost factor
        self.N = N # samples for convergence tests
        self.L = L # levels for convergence tests
        self.Eps = Eps

calltypes = [CallType("European",  4, 500, 5, [0.005, 0.01, 0.02, 0.05, 0.1])]


def opre_gbm(l, N, calltype, randn=numpy.random.randn):
    M = calltype.M # refinement factor

    T   = 1.0  # interval
    r   = 0.05
    sig = 0.2
    K   = 100.0

    nf = M**l
    hf = T/nf

    nc = max(nf/M, 1)
    hc = T/nc

    sums = numpy.zeros(6)

    for N1 in range(1, N+1, 100):
        N2 = min(100, N - N1 + 1)

        X0 = K
        Xf = X0 * numpy.ones(N2)
        Xc = X0 * numpy.ones(N2)

        if l == 0:
            dWf = sqrt(hf) * randn(1, N2)
            Xf[:] = Xf + r*Xf*hf + sig*Xf*dWf

        else:
            for n in range(int(nc)):
                dWc = numpy.zeros((1, N2))

                for m in range(M):
                    dWf = sqrt(hf) * randn(1, N2)
                    dWc[:] = dWc + dWf
                    Xf[:] = (1.0 + r*hf)*Xf + sig*Xf*dWf


                Xc[:] = Xc + r*Xc*hc + sig*Xc*dWc

        if calltype.name == "European":
            Pf = numpy.maximum(0, Xf - K)
            Pc = numpy.maximum(0, Xc - K)
            
        """   
        newPf = numpy.copy(Pf)
        f1 = plt.figure()
        f2 = plt.figure()
        f3 = plt.figure()
        ax1 = f1.add_subplot(111)
        ax1.plot(Xf)
        ax1.set_title(str(N1)+" X level "+str(l))
        ax2 = f2.add_subplot(111)
        ax2.set_title(str(N1)+" P level "+str(l))
        ax2.plot(newPf)
        """
        Pf = numpy.exp(-r*T)*Pf
        Pc = numpy.exp(-r*T)*Pc

        """
        ax3 = f3.add_subplot(111)
        ax3.set_title(str(N1)+" New P level "+str(l))
        ax3.plot(Pf)
        plt.show()
        """

        sums += numpy.array([numpy.sum(Pf - Pc),
                             numpy.sum((Pf - Pc)**2),
                             numpy.sum((Pf - Pc)**3),
                             numpy.sum((Pf - Pc)**4),
                             numpy.sum(Pf),
                             numpy.sum(Pf**2)])

        cost = N*nf # cost defined as number of fine timesteps
        
    return (numpy.array(sums), cost)

if __name__ == "__main__":
    N0 = 1000 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 6  # maximum refinement level

    for (i, calltype) in enumerate(calltypes):
        def opre_l(l, N):
            return opre_gbm(l, N, calltype)

        filename = "opre_gbm%d.txt" % (i+1)
        logfile = open(filename, "w")
        print('\n ---- ' + calltype.name + ' Call ---- \n')
        mlmc_test(opre_l, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile)
        del logfile
        mlmc_plot(filename, nvert=3)
        plt.savefig(filename.replace('.txt', '.eps'))
