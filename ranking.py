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
m_threshold = 19.0

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
        v = Vizier(columns = ['RAJ2000', 'DEJ2000', 'z', 'GWGC', 'BmagHyp', 'ImagHyp', 'Kmag2', 'Jmag2'])

    v.ROW_LIMIT = 99999999
    ra     = np.array(ra)
    dec    = np.array(dec)
    center = SkyCoord(ra.mean(), dec.mean(), unit = (ra_unit, dec_unit))
    width  = abs(ra.max()-ra.min())/2.*ra_unit
    height = abs(dec.max()-dec.min())/2.*dec_unit


    table = v.query_region(center, radius = 1*u.deg, catalog = catalog) # width = width, height = height, catalog = catalog)
    data  = pd.DataFrame()
    # for tablei in table:
    #     data = data.append(tablei.to_pandas(), ignore_index = True)
    data = data.append(table[0].to_pandas())
    return data.dropna()

def get_samples(file, names = ['ra','dec','luminosity_distance']):
    with open(file, 'r') as f:
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

def LD_posterior(LD_s):
    return gaussian_kde(LD_s)


class ranking(cpnest.model.Model):

    def __init__(self, omega):
        self.names=['zgw']
        self.bounds=[[0.025,0.07]]

# 0.7,0.5,0.5,-1.,0.,0.
        self.omega   = omega

    def GalInABox2(self, catalog):
        v = Vizier(columns = ['RAJ2000', 'DEJ2000', 'z'])#, 'GWGC', 'BmagHyp', 'ImagHyp', 'Kmag2', 'Jmag2'])
        v.ROW_LIMIT = 99999999
        center = SkyCoord(M.p_pos.means_[0][0], M.p_pos.means_[0][1], unit = (u.rad, u.rad))
        raggio = np.sqrt(np.diag(M.p_pos.covariances_[0])).max()
        table = v.query_region(center, radius = 5*raggio*u.rad, catalog = catalog) # width = width, height = height, catalog = catalog)
        data  = pd.DataFrame()
        # for tablei in table:
        #     data = data.append(tablei.to_pandas(), ignore_index = True)
        data = data.append(table[1].to_pandas())
        return data.dropna()


    def dropgal(self):
        for i in self.catalog.index:
            if self.pLD(lal.LuminosityDistance(self.omega, self.catalog['z'][i])) < 0.00001:
                self.catalog = self.catalog.drop(i)

    def plot_outputs(self):
        plt.figure(1)
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.xlim([min(M.catalog['RAJ2000'])-0.1, max(M.catalog['RAJ2000'])+0.1])
        plt.ylim([min(M.catalog['DEJ2000'])-0.1, max(M.catalog['DEJ2000'])+0.1])
        S = plt.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], c = M.catalog['p'], marker = '+')
        bar = plt.colorbar(S)
        bar.set_label('p')
        plt.savefig('prob.png')

        # plt.figure(2)
        # S = plt.scatter((M.catalog['BmagHyp']-M.catalog['ImagHyp']),(M.catalog['Jmag2']-M.catalog['Kmag2']), c = M.catalog['p'])
        # plt.colorbar(S)
        # plt.xlabel('B-I')
        # plt.ylabel('J-K')
        # plt.savefig('colorplot.pdf', bbox_inches='tight')

        plt.figure(3)
        S = plt.scatter(M.catalog['RAJ2000'], M.catalog['DEJ2000'], c = M.catalog['ppos'], marker = '+')
        plt.colorbar(S)
        plt.savefig('positionsprob.pdf', bbox_inches = 'tight')


    def log_prior(self,x):
        if not(np.isfinite(super(ranking, self).log_prior(x))):
            return -np.inf
        return 0.

    def log_likelihood(self, x):
        logL = 0.
        zgw = x['zgw']
        # Manca da correggere per il moto proprio (assunto, per ora, gaussiano con errore del 10%)
        # logL = logsumexp([Gaussexp(lal.LuminosityDistance(self.omega, zgi), DL, dDL)+Gaussexp(zgw, zgi, zgi/10.0)+Gaussexp(np.radians(rai), GW.ra.rad, 2.0)+Gaussexp(np.pi/2.0-np.radians(di), GW.dec.rad, 2.0) for zgi,rai,di in zip(self.catalog['z'],self.catalog['RAJ2000'],self.catalog['DEJ2000'])])
        Lh = np.array([gaussian(zgw, zgi, zgi/10.0)*M.pLD(lal.LuminosityDistance(self.omega, zgi))*np.exp(M.p_pos.score_samples([[np.deg2rad(rai),np.deg2rad(di)]])[0])for zgi,rai,di in zip(self.catalog['z'],self.catalog['RAJ2000'],self.catalog['DEJ2000'])])
        logL = np.log(Lh.sum())
        return logL

    def run(self, json_file, show_output = False, run_sampling = True):

        # calcolo posteriors GW
        samples = get_samples(file = json_file)
        self.pLD = gaussian_kde(samples['luminosity_distance'])
        self.p_pos = pos_posterior(samples['ra'],samples['dec'], number = 1)
        probs = []
        self.catalog = self.GalInABox2(catalog='GLADE')
        for ra, dec in zip(self.catalog['RAJ2000'], self.catalog['DEJ2000']):
            probs.append(np.exp(self.p_pos.score_samples([[np.deg2rad(ra),np.deg2rad(dec)]]))[0]) # si riesce ad ottimizzare?
        self.catalog['ppos'] = np.array(probs)

        # Levo le galassie troppo fuori dal posterior
        self.catalog = self.catalog[self.catalog['ppos'] > 50] # empirico!
        # Levo le galassie fuori dal range di distanza
        self.dropgal()
        # run
        job = cpnest.CPNest(self, verbose=1, nthreads=4, nlive=1000, maxmcmc=100)
        if run_sampling:
            job.run()
            posteriors = job.get_posterior_samples(filename = 'posterior.dat')
        # Calcolo posterior su z
        posteriors = np.genfromtxt('posterior.dat', names = True)
        just_z = [post[0] for post in posteriors]
        self.pdfz = gaussian_kde(just_z)

        # Calcolo probabilità e ordino le galassie
        prob = self.catalog['z'].apply(self.pdfz)
        # Non mi piace per niente questa normalizzazione, ma tant'è...
        prob = prob/np.sum(prob)
        self.catalog['p'] = prob
        self.catalog = self.catalog.sort_values('p', ascending = False)
        print('Galaxies:')
        print(self.catalog)
        if show_output:
            self.plot_outputs()

        app = np.linspace(0.01,0.1, 1000)
        plt.figure(4)
        plt.plot(app, self.pdfz(app))
        plt.savefig('pdfz.pdf')

if __name__ == '__main__':

    # show_gaussian_mixture(samples['ra'], samples['dec'], test)
    json_pos = 'posterior_samples.json'
    # Gal_cat = GalInABox([13,15],[-25,-26], u.deg, u.deg, catalog='GLADE')#[::100]
    omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.)
    M = ranking(omega)
    M.run(json_file = json_pos, run_sampling = False, show_output = True)
