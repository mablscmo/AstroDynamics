# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:02:43 2019

@author: monte
"""

import matplotlib.pyplot as plt 
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import numpy.random
from matplotlib.font_manager import FontProperties

font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 35})

class cbin:
    def __init__(self,blim=0.3,parmin=0.5,parmax=1,binN=10,binW=0.2,binmin=-0.2):
        '''Takes data from ASTM13GetData.py, filters it, sorts it into colourbins, plots it, 
        and lets you try try different parameters to see how it changes your values.
        The code should probably have been written as a Jupyter Notebook so I could put in some extra results in an unobtrusive way'''
        #Conversion between (km/s)/kpc and mas/yr
        K = 4.7405
        
        #Filter for the datasets. Datasets obtained from ASTM13GetGaiaData.py
        self.filt=(varpi>10*sigma_varpi)&(abs(b)<np.rad2deg(blim))&(varpi<parmax)&(varpi>parmin)
        
        #Filtered datasets
        colour=BP-RP
        self.colour=colour[self.filt]
        self.l=l[self.filt]
        self.lcos=np.cos(2*np.deg2rad(self.l))
        self.mul=mul[self.filt]
        self.mulK=self.mul*K
        
        #Binning parameters
        self.binN=binN
        self.binW=binW
        self.binmin=binmin
        
    def binning(self,Plot=False,ePlot=False):
        '''Bins data in a range of bins. If Plot==True, plots all the bins and their fits.
        If ePlot==True, plots both the estimates of A and B with errorbars, but also the combined error vs the colour.'''
        values=np.zeros((5,self.binN))
        for i in range(self.binN):
            minlim=self.binmin+self.binW*i
            maxlim=self.binmin+self.binW*(i+1)
            mul=self.mulK[(self.colour<maxlim)&(self.colour>minlim)]
            lcos=self.lcos[(self.colour<maxlim)&(self.colour>minlim)]
            p,cov=np.polyfit(lcos,mul,1,cov=True)
            values[0][i],values[1][i],values[2][i],values[3][i],values[4][i] = p[0],np.sqrt(np.diag(cov)[0]),p[1],np.sqrt(np.diag(cov)[1]),len(lcos)
            if Plot==True:
                plt.figure(figsize=(12,10))    
                plt.plot(lcos,mul,'.')
                plt.plot(lcos,np.polyval(p,lcos))     
                plt.xlabel('cos(2l)')
                plt.ylabel(r'$\mu_l$ $\frac{km/s}{kpc}$')
                plt.title('Fit for colours in range {cmin} - {cmax}'.format(cmin=round(minlim,2),cmax=round(maxlim,2)))
                plt.tight_layout()
        if ePlot==True:
            plt.figure(figsize=(14,10))
            plt.xlabel(r'A $\frac{km/s}{kpc}$')
            plt.ylabel(r'B $\frac{km/s}{kpc}$')
            plt.title('Errors for bin width = {W}'.format(W=self.binW))
            plt.scatter(values[0],values[2],c=range(self.binN),cmap='bwr',s=150)
            plt.errorbar(values[0],values[2],xerr=values[1],yerr=values[3],linestyle='none',color='k',capsize=4)
            plt.tight_layout()
            
            
            
            
            plt.figure(figsize=(10,10))
            plt.xlabel('Colour')
            plt.ylabel('Combined error')
            plt.title('Error for bin width = {W}'.format(W=self.binW))
            plt.plot(np.linspace(self.binmin,self.binmin+self.binW*self.binN,self.binN),values[1]+values[3])
            plt.tight_layout()
            
        return(values)
    
    def BestPlot(self):
        '''Sorts through the bins from self.binning and finds the one with the lowest combined error and plots it.
        This should probably not have been its own function, but the code runs quickly, so I'm not going to fix it.'''
        values=self.binning()
        sumerr=np.ndarray.tolist(values[1]+values[3])
        
        n=sumerr.index(min(sumerr))
        
        minlim = self.binmin + self.binW*n
        maxlim = minlim + self.binW   

        mul=self.mulK[(self.colour<maxlim)&(self.colour>minlim)]
        lcos=self.lcos[(self.colour<maxlim)&(self.colour>minlim)]
        p=np.array([values[0][n],values[2][n]])
        plt.figure(figsize=(12,12))    
        plt.plot(lcos,mul,'.')
        plt.plot(lcos,np.polyval(p,lcos),linewidth=5)
        plt.xlabel('cos(2l)')
        plt.ylabel(r'$\mu_l$ $\frac{km/s}{kpc}$')
        plt.title('Fit for colours in range {cmin} - {cmax}'.format(cmin=round(minlim,2),cmax=round(maxlim,2)))
        plt.tight_layout()
        print(u'Best fit for {A} \u00B1 {Ae}, {B} \u00B1 {Be}. {N} stars in the sample, colourlims=[{cmin},{cmax}]'.format(A=round(p[0],1),Ae=round(values[1][n],3),B=round(values[2][n],1),Be=round(values[3][n],3),N=values[4][n],cmin=round(minlim,2),cmax=round(maxlim,2)))

    def HRplot(self):
        '''Plots the HR-diagram for the filtered data.'''
        Gfilt=G[self.filt]
        Parfilt=varpi[self.filt]
        magG=Gfilt+5*(1+np.log10(Parfilt))
          
        plt.figure(figsize=(16,16))
        plt.hist2d(self.colour,magG,1000,cmin=1,cmap='jet')
        plt.xlim(-2,8)
        plt.ylim(9,28)
        plt.gca().invert_yaxis()
        plt.colorbar(label='Number of stars/bin')
        plt.xlabel('BP-RP colour')
        plt.ylabel(r'$M_G$')
        plt.title(r'HR diagram for filtered sample')
        plt.tight_layout()
        print(len(mulK))    
        
    def FitPlot(self,minlim,maxlim):
        '''Plots the fit for any selected colourinterval.'''
        mul=self.mulK[(self.colour<maxlim)&(self.colour>minlim)]
        lcos=self.lcos[(self.colour<maxlim)&(self.colour>minlim)]   
        p,cov=np.polyfit(lcos,mul,1,cov=True)
        plt.figure(figsize=(12,12))    
        plt.plot(lcos,mul,'.')
        plt.plot(lcos,np.polyval(p,lcos))
        plt.xlabel('cos(2l)')
        plt.ylabel(r'$\mu_l$ $\frac{km/s}{kpc}$')
        plt.title('Fit for colours in range {cmin} - {cmax}'.format(cmin=round(minlim,2),cmax=round(maxlim,2)))        
        plt.tight_layout()
        print(u"Linear fit with {A}\u00B1{Ae}, {B}\u00B1{Be}. {N} stars in the sample, colourlims=[{cmin},{cmax}]".format(A=round(p[0],1),Ae=round(np.sqrt(np.diag(cov)[0]),3),B=round(p[1],1),Be=round(np.sqrt(np.diag(cov)[1]),3),N=len(lcos),cmin=round(minlim,2),cmax=round(maxlim,2)))


