#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import SkyCoord
import astropy.units as u

import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import lal

from scipy import interpolate
from scipy.special import logsumexp
import cpnest, cpnest.model
from scipy.stats import gaussian_kde

from sklearn.mixture import GaussianMixture

'''
Given a GW observation with its posteriors on position and LD, this module
ranks the galaxies localizated within the 95% region according to their probability
of being the host of the compact binary system given a cosmology O.
A = {alpha_i, delta_i, z_i}, X = {alpha_w, delta_w, LD, z_w}.

p(A|X,O) = p(X|A,O)p(A,O)/p(X)

p(X|A,O) = delta(alpha_i-alpha_w)delta(delta_i-delta_w)delta(z_i-z_w)delta(LD-f(z,O)) x
           x p(alpha_w|O)p(delta_w|O)p(LD|O)p(z_i|O)

given that we're making the assumption that the galaxy parameters are exactly known
apart from redshift, where proper motion has to be taken into account.
'''

DL=39.4
dDL=39.4
m_threshold = 19.0

GW = SkyCoord(ra = '13h07m05.49s', dec = '-23d23m02.0s',
            unit=('hourangle','deg'))

def Gaussexp(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)-2.0*np.pi*sigma

def HubbleLaw(D_L, omega): # Da rivedere: test solo 1 ordine
    return D_L*omega.h/(3e3) # Sicuro del numero?

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)

def GalInABox(ra, dec, ra_unit, dec_unit, catalog = 'GLADE2', all = False):
    """
    Given two RA, DEC intervals (thought as the boundaries of a shape), the function
    returns a Pandas DataFrame with all the galaxies in CATALOG which position is within
    the box.
    Parameters
    ----------
    ra, dec: list
        Intervals of angular coordinates within is fully included the desired region.

    ra_unit, dec_unit: astropy.units.core.Unit
        Units of RA and DEC
        Default: deg

    catalog: string, optional.
        Catalog used for the analysis
        Default: GLADE


    all: boolean, optional
        If all = True, returns all the columns downloaded from the catalog.
        Otherwise, only RA, DEC e z.
    Returns
    -------
    df: Pandas DataFrame
        DataFrame Pandas containing all the selected objects.
    """
    if all:
        v = Vizier()
    else:
        v = Vizier(columns = ['RAJ2000', 'DEJ2000', 'zsp2MPZ', 'GWGC', 'BmagHyp', 'ImagHyp', 'Kmag2', 'Jmag2'])

    v.ROW_LIMIT = 99999999
    ra     = np.array(ra)
    dec    = np.array(dec)
    center = SkyCoord(ra.mean(), dec.mean(), unit = (ra_unit, dec_unit))
    width  = (ra.max()-ra.min())/2.*ra_unit
    height = (dec.max()-dec.min())/2.*dec_unit

    table = v.query_region(center, width = width, height = height, catalog = catalog)
    data  = pd.DataFrame()
    # for tablei in table:
    #     data = data.append(tablei.to_pandas(), ignore_index = True)
    data = data.append(table[0].to_pandas())
    print(data)
    print('where?')
    print(data.dropna())
    return data.dropna()

def get_samples(filename, names = ['ra','dec','luminosity_distance']):
    with open(filename, 'r') as f:
        data = json.load(f)

    post = np.array(data['posterior_samples']['SEOBNRv4pHM']['samples'])
    keys = data['posterior_samples']['SEOBNRv4pHM']['parameter_names']

    samples = {}

    for name in names:
        index  = keys.index(name)
        samples[name] = post[:,index]

    return samples

def pos_posterior(ra_s, dec_s, number = 2):
    func = GaussianMixture(n_components = number, covariance_type = 'full')
    samples = []
    for x,y in zip(ra_s, dec_s):
        samples.append(np.array([x,y]))
    func.mean_init = [[0.23,-0.44],[0.4,-0.55]]
    func.fit_predict(samples)
    return func

def show_gaussian_mixture(ra_s, dec_s, mixture):
    x = np.linspace(min(ra_s), max(ra_s))
    y = np.linspace(min(dec_s), max(dec_s))
    X, Y = np.meshgrid(x,y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -mixture.score_samples(XX)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(ra_s,dec_s, 0.8)
    plt.show()

def Galaxies95(boundaries, u_ra = u.rad, u_dec = u.rad, catalog = 'glade2'):
    """
    Dato il contorno (al 95%?) della possibile regione di provenienza di una GW
    restituisce tutte le galassie in un catalogo a scelta che si trovano all'interno
    della suddetta regione.

    NON ANCORA UTILIZZABILE!!!

    Parameters
    ----------
    boundaries: unknown.
        Regione di cielo di forma qualunque proveniente dalla posterior su RA-DEC.

    u_ra, u_dec: astropy.units.core.Unit, optional.
        Unità di misura delle coordinate RA-DEC.

    catalog: string, optional.
        Catalogo dal quale estrarre i dati.
        Default: GLADE2 (pensato per coll. LIGO-Virgo).

    Returns
    -------
    df: Pandas DataFrame
        DataFrame Pandas contenente gli oggetti selezionati dal catalogo.
    """

    ra  = [max(ra_boundaries), min(ra_boundaries)]
    dec = [max(dec_boundaries), min(dec_boundaries)]


    df_all     = GalInABox(ra, dec, u_ra, u_dec, catalog)
# TODO: Scrivere una funzione che data la coppia ra-dec controlli che sia dentro la banana
    clean_data = df_all[df_all[['RAJ2000', 'DEJ2000']].apply(check_in_95, args = (boundaries,)) == True]

    return clean_data

def check_in_95(boundaries, RA, DEC):
    """
    Completamente da scrivere: decidere se un punto è dentro o fuori BOUNDARIES
    [problema: non so in che formato è BOUNDARIES]
    Deve ritornare un Booleano.

    Parameters
    ----------
    boundaries: unknown
        Regione di cielo di forma qualunque proveniente dalla posterior su RA-DEC.

    RA, DEC: float
        Coordinate della galassia (o dell'oggetto) di interesse.

    Returns
    -------
    flag: boolean
        True se l'oggetto è dentro la regione di cielo considerata, False altrimenti.

    """
    if DEC > 1:
        return True
    else:
        return False

class ranking(cpnest.model.Model):

    def __init__(self, catalog, omega):
        self.names=['zgw']
        self.bounds=[[0.001,0.015]]

# 0.7,0.5,0.5,-1.,0.,0.
        self.omega   = omega
        self.catalog = catalog

    def dropgal(self, nsigma = 4):
        for i in self.catalog.index:
            if (self.catalog['zsp2MPZ'][i] > RedshiftCalculation(DL+nsigma*dDL, self.omega)) or (self.catalog['zsp2MPZ'][i] < RedshiftCalculation(DL-nsigma*dDL, self.omega)):
                self.catalog = self.catalog.drop(i)

    def log_prior(self,x):
        if not(np.isfinite(super(ranking, self).log_prior(x))):
            return -np.inf
        return 0.

    def log_likelihood(self, x):
        logL = 0.
        zgw = x['zgw']
        logL = logsumexp([Gaussexp(lal.LuminosityDistance(self.omega, zgi), DL, dDL)+Gaussexp(zgw, zgi, zgi/10.0)+Gaussexp(np.radians(rai), GW.ra.rad, 2.0)+Gaussexp(np.pi/2.0-np.radians(di), GW.dec.rad, 2.0) for zgi,rai,di in zip(self.catalog['zsp2MPZ'],self.catalog['RAJ2000'],self.catalog['DEJ2000'])])
        return logL

    def run(self):
        self.dropgal(nsigma = 6)
        job = cpnest.CPNest(self, verbose=1, nthreads=4, nlive=1000, maxmcmc=1000)
        job.run()
        posteriors = job.get_posterior_samples(filename = 'posteriors.dat')
        just_z = [post[0] for post in posteriors]
        M.pdfz = gaussian_kde(just_z)
        prob = M.catalog['zsp2MPZ'].apply(M.pdfz)
        M.catalog['p'] = prob
        M.catalog = M.catalog.sort_values('p', ascending = False)
        print('Galaxies:')
        print(M.catalog)
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.xlim([min(M.catalog['RAJ2000'])-0.1, max(M.catalog['RAJ2000'])+0.1])
        plt.ylim([min(M.catalog['DEJ2000'])-0.1, max(M.catalog['DEJ2000'])+0.1])
        S = plt.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], c = M.catalog['p'], marker = '+')
        bar = plt.colorbar(S)
        bar.set_label('p')
        plt.savefig('prob.png')



if __name__ == '__main__':

    samples = get_samples('posterior_samples.json')
    test = pos_posterior(samples['ra'],samples['dec'], number = 1)
    # show_gaussian_mixture(samples['ra'], samples['dec'], test)

    Gal_cat = GalInABox([0.10,0.30],[-0.35,-0.55], u.rad, u.rad, catalog='GLADE')#[::100]
    omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.)
    M = ranking(Gal_cat, omega)
    M.dropgal(nsigma = 6)
    print(M.catalog)
    RUN = False
    if RUN:
        M.run()
    posteriors = np.genfromtxt('posterior.dat', names = True)
    M.pdfz = gaussian_kde(posteriors['zgw'])
    prob = M.catalog['zsp2MPZ'].apply(M.pdfz)
    M.catalog['p'] = prob
    M.catalog = M.catalog.sort_values('p', ascending = False)
    print('Galaxies:')
    print(M.catalog)
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.xlim([min(M.catalog['RAJ2000'])-0.1, max(M.catalog['RAJ2000'])+0.1])
    plt.ylim([min(M.catalog['DEJ2000'])-0.1, max(M.catalog['DEJ2000'])+0.1])
    S = plt.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], c = M.catalog['p'], marker = '+')
    bar = plt.colorbar(S)
    bar.set_label('p')
    plt.savefig('prob.png')

    plt.figure(2)
    S = plt.scatter((M.catalog['BmagHyp']-M.catalog['ImagHyp']),(M.catalog['Jmag2']-M.catalog['Kmag2']), c = M.catalog['p'])
    plt.colorbar(S)
    plt.xlabel('B-I')
    plt.ylabel('J-K')
    plt.savefig('colorplot.pdf', bbox_inches='tight')

    probs = []
    for ra, dec in zip(M.catalog['RAJ2000'], M.catalog['DEJ2000']):
        probs.append(np.exp(test.score_samples([[np.deg2rad(ra),np.deg2rad(dec)]]))[0])

    M.catalog['ppos'] = np.array(probs)

    M.catalog['ppos'] = M.catalog[M.catalog['ppos'] > 100] # empirico! 
    plt.figure(3)

    S = plt.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], c = M.catalog['ppos'])
    plt.colorbar(S)
    plt.savefig('positionsprob.pdf', bbox_inches = 'tight')
