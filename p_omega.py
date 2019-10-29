#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from setup_env import *
#from mmlibrary import *

from astropy.coordinates import SkyCoord
import astropy.units as u

from mmlibrary import *

import numpy as np
import lal

from scipy.special import logsumexp
import cpnest, cpnest.model

# Oggetto per test: GW170817
#GW = SkyCoord('13h07m05.49s', '23d23m02.0s', unit=(u.hourangle, u.deg))
DL=33.4
dDL=3.34

GW = SkyCoord(ra = '13h07m05.49s', dec = '23d23m02.0s',
            unit=('hourangle','deg'))


def Mstar(omega):
    '''
    Calcolo magnitudine di taglio Schechter function
    '''
    return -20.47 + 5.0*np.log10(omega.h)

def Schechter_unnormed(M, omega, alpha):
    '''
    Funzione di Schechter non normalizzata
    '''
    Ms = Mstar(omega)
    tmp = 10**(-0.4*(M-Ms))
    return tmp**(alpha+1.0)*np.exp(-tmp)

def normalise(omega, alpha, Mmin = -30,Mmax = -10):
    '''
    Normalizzazione funzione di Schechter (todo: fare analitica)
    '''
    M = np.linspace(Mmin, Mmax, 100)
    return np.sum([Schechter_unnormed(Mi, omega, alpha = alpha)*np.diff(M)[0] for Mi in M])

def Schechter(M, omega, alpha = -1.07):
    '''
    Funzione di Schechter normalizzata
    '''
    return Schechter_unnormed(M, omega, alpha = alpha)/normalise(omega, alpha = alpha)

def Mthreshold(DL, mth = 27.0):
    '''
    Magnitudine assoluta di soglia
    '''
    return mth - 5.0*np.log10(1e5*DL)

def mabs(m, DL):
    return m - 5.0*np.log10(1e5*DL)


def HubbleLaw(D_L, omega): # Da rivedere: test solo 1 ordine
    return D_L*omega.h/(3e3) # Sicuro del numero?

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

class completeness(cpnest.model.Model):

    def __init__(self, catalog):
        self.names=['z', 'h', 'om', 'ol']
        self.bounds=[[0.001,0.012],
                    [0.5,1.],
                    [0.04,1.],
                    [0.,1.]]
        self.omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.,0.,0.)
        self.catalog = catalog


    def log_prior(self, x):
        # controllo finitezza e theta(M-Mth)

        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        else:
            self.omega.h = x['h']
            self.omega.om = x['om']
            self.omega.ol = x['ol']
            zgw  = x['z']
            logP = 0.0
            for zi,mi in zip(self.catalog['z'],self.catalog['Bmag']):
                DL = lal.LuminosityDistance(self.omega, zi)
                Mabsi = mabs(mi,DL)
                if  Mthreshold(DL) < Mabsi:

                    return -np.inf
                else:
                    # Update parametri cosmologici con simulazione

                    # Calcolo prior. Ciascuna coordinata è pesata con le probabilità
                    # delle coordinate ('banane') GW, così come z.
                    # Temporaneamente, è assunta gaussiana intorno a un evento.
                    logP += np.log(Schechter(Mabsi, self.omega))
                    #log_P_RA     = np.log(gaussian(x['ra'],Gal.ra.rad,Gal.ra.rad/100.))
                    #log_P_DEC    = np.log(gaussian(x['dec'],Gal.dec.rad,Gal.dec.rad/100.))
                    logP += np.log(lal.ComovingVolumeElement(zi, self.omega))

            return logP
                    # PROBLEMA! Come introduco le delta(ra,dec)?

    def log_likelihood(self, x):
        logL = 0.0
        zgw  = x['z']

        logL += np.log(gaussian(lal.LuminosityDistance(self.omega, zgw), DL,dDL))
        logL += logsumexp([gaussian(zgw, zgi, zgi/10.0) for zgi in self.catalog['z']])
        #logL += np.log(gaussian(x['ra'],GW.ra.rad,GW.ra.rad/10.))
        #logL += np.log(gaussian(x['dec'],GW.dec.rad,GW.dec.rad/10.))

        return logL

if __name__ == '__main__':
    Gal_cat = GalInABox([190,200],[-25,-15], u.deg, u.deg, catalog='GLADE')[::100]
    M = completeness(Gal_cat)

    job = cpnest.CPNest(M, verbose=2, nthreads=4, nlive=1000, maxmcmc=1024)
    job.run()
# GLADE galaxy catalog
