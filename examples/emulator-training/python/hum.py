#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:45:02 2019

@author: puk
"""

import numpy as np
# Description:
# conversions between different humidity quantities

Mw=18.0160 # molecular weight of water
Md=28.9660 # molecular weight of dry air
R =  8.31432E3 # gas constant
Rd = R/Md # specific gas constant for dry air
Rv = R/Mw # specific gas constant for vapour
Lv = 2.5e6 # heat release for condensation of water vapour [J kg-1]
eps = Mw/Md
#saturation pressure
def esat(T):
    ''' get sateration pressure (units [Pa]) for a given air temperature (units [K])'''
    TK = 273.15
    e1 = 101325.0
    logTTK = np.log10(T/TK)
    esat =  e1*10**(10.79586*(1-TK/T)-5.02808*logTTK+ 1.50474*1e-4*(1.-10**(-8.29692*(T/TK-1)))+ 0.42873*1e-3*(10**(4.76955*(1-TK/T))-1)-2.2195983) 
    return esat
def esat2(T):
    ''' a simpler form for the saturation pressure (units [Pa]) for a given air temperature (units [K]), based on clausius-claperyon'''
    return 611.*np.exp(-Lv/Rv*(1./T - 1./273.16))

def rh2mixr(RH,p,T):
    '''purpose: conversion relative humidity (unitless) to mixing ratio [kg/kg]'''
    es = esat(T)
    return Mw/Md*RH*es/(p-RH*es)

def mixr2rh(mixr,p,T):
    '''purpose: conversion mixing ratio to relative humidity [kg/kg] (not tested)'''
    return mixr * p/((mixr+Mw/Md)*esat(T))

def mixr2sh(W):
    '''conversion from mixing ratio (units [kg/kg]) to specific humidity (units also [kg/kg])
    '''
    return W/(1.+W)
def sh2mixr(qv):
    '''conversion from specific humidity (units [kg/kg]) to mixing ratio (units also [kg/kg])
    '''
    return qv/(1.-qv)
def ea2mixr(P,ea):
    '''conversion from dry air and vapour pressure (units [Pa]) to mixing ratio (units [kg/kg]) 
    '''
    return 0.622*ea/(P-ea)
# def mixr2ea(P,mixr):
#     '''conversion from dry air pressure (units [Pa]) and mixing ratio (units [kg/kg] to vapour pressure (units [Pa])
#     '''
#     return P*mixr/(0.622+ea)

def ah2mixr (rhov,p,T):
    '''conversion from absolute humidity (units [kg/m**3]) to mixing ratio (units also [kg/kg])
       not tested
    '''
    return (Rd * T)/(p/rhov-Rv*T)

# not tested
def wvap2sh(e,p):
    '''conversion from water vapour pressure (units [Pa]) to specific humidity (units [kg/kg]) 
    '''
    return eps*e/(p-(1.-eps)*e)

# def rh2ah(RH,p,T):
#     '''conversion relative humidity to absolute humidity (kg Water per m^3 Air)'''
#     mixr=rh2mixr(RH, p,T)
#     sh=mixr2sh(mixr)
#     return  sh*rhov(T,p,sh) 