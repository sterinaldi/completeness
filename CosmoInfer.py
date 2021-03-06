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
from scipy.integrate import dblquad

# Oggetto per test: GW170817
#GW = SkyCoord('13h07m05.49s', '23d23m02.0s', unit=(u.hourangle, u.deg))
DL=33.4
dDL=3.34
m_threshold = 19.0

GW = SkyCoord(ra = '13h07m05.49s', dec = '23d23m02.0s',
            unit=('hourangle','deg'))

def Gaussexp(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)-2.0*np.pi*sigma

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

def normalise(omega, alpha = -1.07, Mmin = -30, Mmax = 0):
    '''
    Normalizzazione funzione di Schechter (todo: fare analitica)
    '''
    M = np.linspace(Mmin, Mmax, 64)
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

def normalise_integrand(M, z, omega, K):
    PS = Schechter(M, omega)
    PV = lal.ComovingVolumeElement(z, omega)
#    print(" ==== integrand ===",PS,PV,K,PS*PV,(PS*PV)**K)
    return (PS*PV)**K

def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Calcola per ricorsione il redshift data una certa luminosità.
    Variando il parametro limit si ritocca il numero di cifre significative.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)

class completeness(cpnest.model.Model):
    """
    p(O|W(G+~G)) = p(O) p(W(G+~G)|O)/(p(W(G+~G))
    p(W(G+~G)|O) = p(WG)|O)+p(W~G|O) = p(W|GO)p(G|O)+p(W|~GO)p(~G|O)
                 = p(W|GO)p(G|O)+p(W|~GO)(1-p(G|O))
    """
    def __init__(self, catalog):
        self.names=['zgw', 'h', 'om', 'ol', 'mgal', 'zgal', 'alpha', 'delta']
        self.bounds=[[0.001,0.015],
                    [0.5,2.0],
                    [0.04,1.0],
                    [0.0,1.0],
                    [m_threshold,30.0],
                    [0.001,0.015],
                    [GW.ra.rad-0.1,GW.ra.rad+0.1],
                    [GW.dec.rad-0.1,GW.dec.rad+0.1]]

        self.omega = lal.CreateCosmologicalParameters(0.7,0.5,0.5,-1.,0.,0.)
        self.catalog = catalog

    def dropgal(self, nsigma = 4):
        for i in self.catalog.index:
            if (self.catalog['z'][i] > RedshiftCalculation(DL+nsigma*dDL, self.omega)) or (self.catalog['z'][i] < RedshiftCalculation(DL-nsigma*dDL, self.omega)):
                self.catalog = self.catalog.drop(i)

    def log_prior(self,x):
        if not(np.isfinite(super(completeness, self).log_prior(x))):
            return -np.inf
        return 0.0

    def log_prob_detected_galaxies(self, x):
        # controllo finitezza e theta(M-Mth)

        self.omega.h = x['h']
        self.omega.om = x['om']
        self.omega.ol = x['ol']
        zgw  = x['zgw']
        logP = 0.0
        Vmax = lal.ComovingVolume(self.omega, self.bounds[0][1])
        for zi,mi in zip(self.catalog['z'],self.catalog['Bmag']):
            DL = lal.LuminosityDistance(self.omega, zi)
            Mabsi = mabs(mi,DL)
            if  Mthreshold(DL) < Mabsi:
                return -np.inf
            else:
                logP += np.log(Schechter(Mabsi, self.omega))
                logP += np.log(lal.ComovingVolumeElement(zi, self.omega)/Vmax)

        return logP

    def log_prob_non_detected_galaxies(self, x):

        # controllo finitezza e theta(M-Mth)

        self.omega.h = x['h']
        self.omega.om = x['om']
        self.omega.ol = x['ol']
        zgal  = x['zgal']
        mgal  = x['mgal']
        logP = 0.0

        DL = lal.LuminosityDistance(self.omega, zgal)
        Mth = Mthreshold(DL)
        Mabsi = mabs(mgal,DL)

        if  Mthreshold(DL) > Mabsi:
            return -np.inf
        else:
            # compute the allowed volume for the source
            gDL=33.4
            gdDL=3.34
            V = (4./3.)*np.pi*(gDL-gdDL)**3
            # find the number density of galaxies from the low end of the Schecter
            # distribution
            n0 = 1./normalise(self.omega, Mmax = Mth)
            K = np.int(V*n0)-len(self.catalog)

            if K <= 0.0:
                # no galaxies are missing
                return -np.inf

            logP += np.log(Schechter(Mabsi, self.omega))
            logP += np.log(lal.ComovingVolumeElement(zgal, self.omega))
#            norm = np.log(dblquad(normalise_integrand, self.bounds[0][0], self.bounds[0][1],
#                      lambda x: mabs(mgal,lal.LuminosityDistance(self.omega, self.bounds[4][0])),
#                      lambda x: mabs(mgal,lal.LuminosityDistance(self.omega, self.bounds[4][0])),
#                      args = (self.omega, 1, ))[0])
            # since for alpha < 0.0 the Schecter distribution diverges, approximate the integral with the
            # maximum of the integral
            norm = K*(np.log(Schechter(mabs(self.bounds[4][1],lal.LuminosityDistance(self.omega, self.bounds[0][1])),self.omega))+np.log(lal.ComovingVolumeElement(self.bounds[0][1], self.omega)))

        return K*(logP-norm)

    def log_likelihood(self, x):
        # non detected
        logL_non_detected = 0.0
        zgal  = x['zgal']
        log_p_non_det = self.log_prob_non_detected_galaxies(x)
        if np.isinf(log_p_non_det):
            logL_non_detected = -np.inf
        else:
            logL_non_detected += Gaussexp(lal.LuminosityDistance(self.omega, zgal), DL, dDL)
            logL_non_detected += Gaussexp(x['alpha'], GW.ra.rad, 1.0/10.0)
            logL_non_detected += Gaussexp(x['delta'], GW.dec.rad, 1.0/10.0)
            logL_non_detected += log_p_non_det

        # detected
        logL_detected = 0.0
        zgw  = x['zgw']
        # estimate the probability of detecting a galaxy as 1-prob_non_detection
        log_p_det = logsumexp([0.0,log_p_non_det],b=[1,-1])

        logL_detected += Gaussexp(lal.LuminosityDistance(self.omega, zgw), DL, dDL)
        logL_detected += logsumexp([Gaussexp(zgw, zgi, zgi/10.0)+Gaussexp(np.radians(rai), GW.ra.rad, 1.0/10.0)+Gaussexp(np.pi/2.0-np.radians(di), GW.dec.rad, 1.0/10.0) for zgi,rai,di in zip(self.catalog['z'],self.catalog['RAJ2000'],self.catalog['DEJ2000'])])

        logL_detected += log_p_det
        logL = logsumexp([logL_detected,logL_non_detected])

        return logL

if __name__ == '__main__':

    Gal_cat = GalInABox([190,200],[-22,-17], u.deg, u.deg, catalog='GLADE')#[::100]
    M = completeness(Gal_cat)
#    NGC4993 = Vizier.query_object('NGC4993', catalog = 'GLADE')[1].to_pandas()
#    M = completeness(NGC4993)
    # M.dropgal(nsigma = 3)
    # print([zi for zi in M.catalog['z']])
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # S = ax.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], M.catalog['z'], c=M.catalog['Bmag'])
    # plt.colorbar(S)
    # plt.show()
    # exit()
#    print (M.catalog)
#    exit()
    job = cpnest.CPNest(M, verbose=2, nthreads=4, nlive=5000, maxmcmc=1000)
    job.run()
