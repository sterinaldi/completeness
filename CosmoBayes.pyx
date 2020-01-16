#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Come importo il file ../../cosmolisa/cosmology.c?
(contiene lal wrappato)
'''
from __future__ import division

from astropy.coordinates import SkyCoord
import astropy.units as u

from mmlibrary import *

import numpy as np
cimport numpy as np
import lal

cimport cython

from scipy.special import logsumexp
import cpnest, cpnest.model
from scipy.integrate import dblquad

cpdef double Gaussexp(double x, double mu, double sigma):
    '''
    Gaussian exponent
    '''
    return -(x-mu)**2/(2*sigma**2)-2.0*np.pi*sigma

cpdef double Mthreshold(double DL, double mth = 24.0):
    '''
    Absolute treshold magnitude
    '''
    return mth - 5.0*np.log10(1e5*DL)

cpdef double mrel(M, DL):
    '''
    Apparent magnitude
    '''
    return M + 5.0*np.log10(1e5*DL)

cpdef gaussian(x,x0,sigma):
    '''
    self-explaining.
    '''
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
