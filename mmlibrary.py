#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:22:26 2019

@author: Stefano Rinaldi (s.rinaldi9@studenti.unipi.it)

Il presente modulo è una raccolta di funzioni, in continuo aggiornamento,
pensate per il corso di Multimessenger Physics Laboratory.
La libreria è in continuo aggiornamento, di pari passo con l'avanzare del corso.

Il presente modulo è fornito "as it is", senza alcun tipo di supporto tecnico.
Gli autori declinano ogni responsabilità derivante dall'uso improprio del codice.

Sono richieste le seguenti librerie:
    - astropy (http://www.astropy.org)
    - pandas (https://pandas.pydata.org)

Contenuto della liberia:

    - CLASS GWevent()
    - GalInABox()

"""

import numpy as np
import pandas as pd

from astropy.table import Table
from numpy import empty, zeros

from astropy.coordinates import SkyCoord as sc
from astropy.coordinates import EarthLocation
from astropy.time import Time as tm
from astropy import units as u
from numpy import arccos, deg2rad, e, log10, pi, sqrt, tan

from astroquery.vizier import Vizier

class GWevent():
    """
__________________

Classe contenente informazioni su singola detection gravitazionale.
Attenzione: l'incertezza sulla posizione è contenuta in un oggetto SkyCoord.
self.err_coord è quindi da intendersi come un vettore di errori su self.coord
piuttosto che come una seconda posizione nel cielo.

==================

Richiesti:

# - Luminosity Distance (con incertezza)
# - coordinate RA-DEC (con incertezza e udm - per SkyCoord)

    """
    def update_position(self):
        skycoordinates = sc(self.ra, self.dec, unit=(self.u_ra, self.u_dec))
        err            = sc(self.dra, self.ddec, unit=(self.u_ra, self.u_dec))
        self.coord     = skycoordinates
        self.err_coord = err
        # Promemoria: questo non è proprio il modo giusto di usare SkyCoord ma serve
        # per avere un convertitore rapido.

    def __init__(self, **kwargs):
        allowed_keys = set(['LD', 'dLD' , 'ra', 'dra', 'dec', 'ddec', 'u_ra', 'u_dec', 'name'])
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, value) for key,value in kwargs.items() if key in allowed_keys)
        self.update_position()

def GalInABox(ra, dec, ra_unit, dec_unit, catalog = 'GLADE', all = False):
    """
    Dati gli intervalli RA e DEC (pensati come gli estremi di una forma qualunque),
    la funzione restituisce un DataFrame contenente tutte le galassie contenute
    in CATALOG entro il rettangolo definito dagli intervalli

    Parameters
    ----------
    ra, dec: list
        Intervalli di coordinate angolari entro le quali è interamente contenuta
        la regione desiderata

    ra_unit, dec_unit: astropy.units.core.Unit
        Unità di misura in cui sono espressi RA e DEC.
        Default: deg

    catalog: string, optional.
        Catalogo dal quale estrarre i dati.
        Default: GLADE2 (pensato per coll. LIGO-Virgo).


    all: boolean, optional
        Se all = True restituisce tutte le colonne scaricate dal catalogo.
        Se all = False, solamente quelle corrispondenti a RA, DEC e z.
    Returns
    -------
    df: Pandas DataFrame
        DataFrame Pandas contenente gli oggetti selezionati dal catalogo.
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
