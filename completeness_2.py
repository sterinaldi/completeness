#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from setup_env import *
#from mmlibrary import *

from astropy.coordinates import SkyCoord
import astropy.units as u

from mmlibrary import *

import numpy as np
import lal

import cpnest, cpnest.model

# Oggetto per test: GW170817
#GW = SkyCoord('13h07m05.49s', '23d23m02.0s', unit=(u.hourangle, u.deg))
DL=33.4

GW = SkyCoord(ra = '13h07m05.49s', dec = '23d23m02.0s',
            unit=('hourangle','deg'))


Gal = SkyCoord('13h07m05.47s', '23d23m02.06s', unit=(u.hourangle, u.deg))
Gal.z=0.009787

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

def Mthreshold(z, omega, mth = 24):
    '''
    Magnitudine assoluta di soglia
    '''
    return mth - 5.0*np.log10(1e5*lal.LuminosityDistance(omega, z))


def HubbleLaw(D_L, omega): # Da rivedere: test solo 1 ordine
    return D_L*omega.h/(3e3) # Sicuro del numero?

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

class completeness(cpnest.model.Model):

    def __init__(self):
        self.names=['z', 'ra', 'dec', 'M', 'h', 'om', 'ol']
        self.bounds=[[0.00001,0.2],
                    [0., 2.*np.pi],
                    [-np.pi/2., np.pi/2.],
                    [-30,-10],
                    [0.5,1.],
                    [0.04,1.],
                    [0.,1.]]
        self.omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.,0.,0.)

    def log_prior(self, x):
        # controllo finitezza e theta(M-Mth)
        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        else:
            if  x['M'] > Mthreshold(x['z'], self.omega):
                return -np.inf
            else:
                # Update parametri cosmologici con simulazione
                self.omega.h = x['h']
                self.omega.om = x['om']
                self.omega.ol = x['ol']
                # Calcolo prior. Ciascuna coordinata è pesata con le probabilità
                # delle coordinate ('banane') GW, così come z.
                # Temporaneamente, è assunta gaussiana intorno a un evento.
                log_P_Z      = np.log(gaussian(x['z'], Gal.z, Gal.z/10.0))
                log_P_S      = np.log(Schechter(x['M'], self.omega))
                log_P_RA     = np.log(gaussian(x['ra'],Gal.ra.rad,Gal.ra.rad/100.))
                log_P_DEC    = np.log(gaussian(x['dec'],Gal.dec.rad,Gal.dec.rad/100.))
                log_P_ComVol = np.log(lal.ComovingVolumeElement(x['z'], self.omega))
                return log_P_S+log_P_Z+log_P_ComVol+log_P_RA+log_P_DEC
                # PROBLEMA! Come introduco le delta(ra,dec)?

    def log_likelihood(self, x):
        logL = 0.0
        logL += np.log(gaussian(lal.LuminosityDistance(self.omega, x['z']), 33.4,3.34))
        logL += np.log(gaussian(x['ra'],GW.ra.rad,GW.ra.rad/10.))
        logL += np.log(gaussian(x['dec'],GW.dec.rad,GW.dec.rad/10.))
        return logL

if __name__ == '__main__':
    M = completeness()
    job = cpnest.CPNest(M, verbose=2, nthreads=8, nlive=2000, maxmcmc=1024)
    job.run()
# GLADE galaxy catalog
