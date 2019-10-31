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

def Gaussexp(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)

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

def Mthreshold(DL, mth = 24.0):
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
        self.names=['zgw', 'h', 'om', 'ol', 'mgal', 'zgal', 'alpha', 'delta']
        self.bounds=[[0.001,0.012],
                    [0.5,1.],
                    [0.04,1.],
                    [0.,1.],
                    [24.0,30.0],
                    [0.001,0.012],
                    [GW.ra.rad-0.1,GW.ra.rad+0.1],
                    [GW.dec.rad-0.1,GW.dec.rad+0.1]]

        self.omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.,0.,0.)
        self.catalog = catalog

    def dropgal(self):
        for i in self.catalog.index:
            if abs(lal.LuminosityDistance(self.omega, self.catalog['z'][i])- DL) > 4*dDL:
                self.catalog = self.catalog.drop(i)

    def log_prior(self,x):
        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        return 0.0

    def log_prob_detected_galaxies(self, x):
        # controllo finitezza e theta(M-Mth)

        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        else:
            self.omega.h = x['h']
            self.omega.om = x['om']
            self.omega.ol = x['ol']
            zgw  = x['zgw']

            logP = 0.0
            for zi,mi in zip(self.catalog['z'],self.catalog['Bmag']):
                DL = lal.LuminosityDistance(self.omega, zi)
                Mabsi = mabs(mi,DL)
                if  Mthreshold(DL) < Mabsi:
                    return -np.inf
                else:
                    logP += np.log(Schechter(Mabsi, self.omega))
                    logP += np.log(lal.ComovingVolumeElement(zi, self.omega))

            return logP

    def log_prob_non_detected_galaxies(self, x):
        # controllo finitezza e theta(M-Mth)

        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        else:
            self.omega.h = x['h']
            self.omega.om = x['om']
            self.omega.ol = x['ol']
            zgal  = x['zgal']
            mgal  = x['mgal']
            logP = 0.0
            K = 380
            DL = lal.LuminosityDistance(self.omega, zgal)
            Mabsi = mabs(mgal,DL)
            if  Mthreshold(DL) > Mabsi:
                print(Mthreshold(DL),Mabsi)
                return -np.inf
            else:
                logP += np.log(Schechter(Mabsi, self.omega))
                logP += np.log(lal.ComovingVolumeElement(zgal, self.omega))

            return K*logP
            
    def log_likelihood(self, x):
        logL_detected = 0.0
        zgw  = x['zgw']
        log_p_det = self.log_prob_detected_galaxies(x)
        if np.isinf(log_p_det):
            print('failed 1!')
            return -np.inf
        # detected
        logL_detected += np.log(gaussian(lal.LuminosityDistance(self.omega, zgw), DL,dDL))
        logL_detected += logsumexp([Gaussexp(zgw, zgi, zgi/10.0)+Gaussexp(ai, GW.ra.rad, GW.ra.rad/10.0)+Gaussexp(di, GW.dec.rad, GW.dec.rad/10.0) for zgi,ai,di in zip(self.catalog['z'],self.catalog['RAJ2000'],self.catalog['DEJ2000'])])
        logL_detected += log_p_det
        
        # non detected
        logL_non_detected = 0.0
        zgal  = x['zgal']
        log_p_non_det = self.log_prob_non_detected_galaxies(x)
        if np.isinf(log_p_non_det):
            print('failed 2!')
            return -np.inf

        logL_non_detected += np.log(gaussian(lal.LuminosityDistance(self.omega, zgal), DL,dDL))
        logL_non_detected += np.log(gaussian(x['alpha'], GW.ra.rad, GW.ra.rad/10.0))
        logL_non_detected += np.log(gaussian(x['delta'], GW.dec.rad, GW.dec.rad/10.0))
        logL_non_detected += log_p_non_det
        logL = logsumexp([logL_detected,logL_non_detected])
#        print(logL,logL_detected,logL_non_detected,log_p_det,log_p_non_det)
        return logL

if __name__ == '__main__':
    #Gal_cat = GalInABox([190,200],[-25,-15], u.deg, u.deg, catalog='GLADE')[::100]

    NGC4993 = Vizier.query_object('NGC4993', catalog = 'GLADE')[1].to_pandas()

    M = completeness(NGC4993)
##    M.dropgal()
#    print (M.catalog)
#    exit()
    job = cpnest.CPNest(M, verbose=2, nthreads=4, nlive=1000, maxmcmc=100)
    job.run()
#    N = completeness_notG()
#
#    job_G = cpnest.CPNest(M, verbose=2, nthreads=4, nlive=1000, maxmcmc=1024)
#    job_G.run()
#    job_notG = cpnest.CPNest(N, verbose=2, nthreads=4, nlive=1000, maxmcmc=1024)
#    job_notG.run()

# GLADE galaxy catalog
