#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import SkyCoord
import astropy.units as u


import numpy as np
import lal

from scipy.special import logsumexp
import cpnest, cpnest.model

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

DL=33.4
dDL=3.34
m_threshold = 19.0

GW = SkyCoord(ra = '13h07m05.49s', dec = '23d23m02.0s',
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

def GalInABox(ra, dec, ra_unit, dec_unit, catalog = 'GLADE', all = False):
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
        v = Vizier(columns = ['RAJ2000', 'DEJ2000', 'z', 'Bmag'])

    v.ROW_LIMIT=-1
    ra     = np.array(ra)
    dec    = np.array(dec)
    center = sc(ra.mean(), dec.mean(), unit = (ra_unit, dec_unit))
    width  = (ra.max()-ra.min())/2.*ra_unit
    height = (dec.max()-dec.min())/2.*dec_unit

    table = v.query_region(center, width = width, height = height, catalog = catalog)
    data  = pd.DataFrame()
    for tablei in table:
        data = data.append(tablei.to_pandas(), ignore_index = True)

    return data.dropna()

def Galaxies95(boundaries, u_ra = u.rad, u_dec = u.rad, catalog = 'GLADE'):
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
