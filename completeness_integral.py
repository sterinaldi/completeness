#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import lal

import cpnest, cpnest.model


def Mstar(omega):
    return -20.47 + 5.0*np.log10(omega.h)

def Schechter_unnormed(M, omega, alpha = -1.07):
    Ms = Mstar(omega)
    tmp = 10**(-0.4*(M-Ms))
    return tmp**(alpha+1.0)*np.exp(-tmp)

def normalise(omega, alpha = -1.07, Mmin = -30,Mmax = -10):
    M = np.linspace(Mmin, Mmax, 100)
    return np.sum([Schechter_unnormed(Mi, omega)*np.diff(M)[0] for Mi in M])

def Schechter(M, omega, alpha = -1.07):
    return Schechter_unnormed(M, omega, alpha = alpha)/normalise(omega, alpha = alpha)

def Mthreshold(z, omega, mth = 24):
    return mth - 5.0*np.log10(1e5*lal.LuminosityDistance(omega, z))


class completeness(cpnest.model.Model):

    def __init__(self):
        self.names=['redshift','magnitude','h','om','ol']#,u'ra',u'dec'
        self.bounds=[[0.0,2],
                     [-30,-10],
                     [0.5,1.0],
                     [0.04,1.0],
                     [0.0,1.0]]

        self.omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.0,0.0,0.0)

    def log_prior(self, x):
        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        else:

            self.omega.h = x['h']
            self.omega.om = x['om']
            self.omega.ol = x['ol']
            logP = np.log(lal.ComovingVolumeElement(x['redshift'], self.omega))

            return logP

    def log_likelihood(self, x):
        if x['magnitude'] > Mthreshold(x['redshift'], self.omega):
            return -np.inf
        else:
            return np.log(Schechter(x['magnitude'], self.omega))

if __name__ == "__main__":
#    import matplotlib.pyplot as plt
#    omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.0,0.0,0.0)
#    M = np.linspace(-30.0, -10.0, 100)
#    print (np.sum([Schechter(Mi, omega)*np.diff(M)[0] for Mi in M]))
#    plt.plot(M, Schechter(M, omega))
#    plt.show()
#    exit()

#    import matplotlib.pyplot as plt
#    omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.0,0.0,0.0)
#    z = np.linspace(0.001,1.0,100)
#    plt.plot(z, [Mthreshold(zi, omega, mth = 17) for zi in z])
#    plt.show()
#    exit()
#
    M = completeness()
    job = cpnest.CPNest(M,verbose=2,nthreads=4,nlive=5000,maxmcmc=1024)
    job.run()
