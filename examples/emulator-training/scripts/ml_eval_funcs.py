#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:01:44 2021

@author: puk
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))

def plot_hist2d(y_true,y_pred,nbins,norm):
  
    x = y_true.flatten()
    y = y_pred.flatten()
    err =  np.corrcoef(x,y)[0,1]; err = err**2
    if norm == True:
        fig, ax = plt.subplots()
        (counts, ex, ey, img) = ax.hist2d(x, y, bins=nbins, norm=LogNorm())
    else:
        plt.hist2d(x, y, bins=40)
    if (np.max(x) < 1.1) & (np.min(x) > -0.1):
        plt.xlabel('Transmittance')
        plt.ylabel('Transmittance (predicted)')
    elif (np.min(x) < 0.0):
        plt.xlabel('Normalized optical depth')
        plt.ylabel('Normalized optical depth (predicted)')
    else:
        plt.xlabel('Optical depth')
        plt.ylabel('Optical depth (predicted)')
        
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    ax.set_xlim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    ax.set_ylim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')

    mse_err = mse(x,y)
    mae_err = np.mean(np.abs((x - y)))
    textstr0 = 'R-squared = {:0.5f}'.format(err)
    textstr1 = 'MSE = {:0.5f}'.format(mse_err)
    textstr2 = 'MAE = {:0.5f}'.format(mae_err)
    plt.annotate(textstr0, xy=(-7.0, 0.15), xycoords='axes fraction')
    plt.annotate(textstr1, xy=(-7.0, 0.10), xycoords='axes fraction')
    plt.annotate(textstr2, xy=(-7.0, 0.05), xycoords='axes fraction')
    ax.grid()
    ident = [xmin, xmax]
    ax.plot(ident,ident,'k')
    del x,y


def plot_hist2d_tau(y_true,y_pred,nbins,norm):
    x = y_true[(y_true<10)&(y_pred<10)].flatten()
    y = y_pred[(y_true<10)&(y_pred<10)].flatten()
    plot_hist2d(x,y,nbins,norm)
    del x,y

def plot_hist2d_T(y_true,y_pred,nbins,norm):
    y_true = np.exp(-y_true).flatten()
    y_pred = np.exp(-y_pred).flatten()
    plot_hist2d(y_true,y_pred,nbins,norm)
    del y_true,y_pred