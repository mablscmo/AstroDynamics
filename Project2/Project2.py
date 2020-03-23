# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:59:59 2019

@author: monte
"""
import matplotlib.pyplot as plt 
import numpy as np
import numpy.random
from matplotlib.font_manager import FontProperties
import csv

with open('P2Data.csv', 'w', newline='') as file:
    writer=csv.writer(file)
    writer.writerow(["$C_{min}$","$C_{max}$","$C_{mean}$","$T_{mean}$","$M_{G min}$","$p_{min}$","$N_{stars}$","$u_{avg}$","$v_{avg}$","$w_{avg}$","D11","D12","D13","D22","D23","D33"])

K = 4.7405

colour = BP-RP
MAG = G + 5*(1 + np.log10(varpi))
index=0

def test(Cmin=-0.2,step=0.05):
    cmin=round(Cmin,2)
    cmax=round(Cmin+step,2)


    msx=[-0.274,-0.202,-0.142,-0.066,-0.0032,0.0291,0.262,0.401,0.500,0.575,0.682,0.749,0.795]
    msy=[14.00,14.74,15.38,16.14,16.54,16.76,17.56,17.89,18.15,18.58,19.32,19.71,20.02]
    msX=[-0.276,-0.0538,0.130,0.258,0.392,0.532,0.642,0.735,0.797]
    msY=[12.04,12.99,13.68,13.96,14.41,14.88,15.22,15.71,16.06]


    #Filter for the datasets. Datasets obtained from ASTM13GetGaiaData.py
    #Filters away high error stars, sets edges for colourbins, removes far-away stars with 
    #possibly negative parallaxes
    filt=(varpi>10*sigma_varpi)&(colour>cmin)&(colour<cmax)&(varpi>0.1)
        
    colourf=colour[filt]
    lf=np.deg2rad(l[filt])
    bf=np.deg2rad(b[filt])
    pf=varpi[filt]
    mulf=mul[filt]
    mubf=mub[filt]
    gf=G[filt]
    Gf=gf+5*(1+np.log10(pf))

    #Creating filter to isolate the main sequence based on points measured in the HR diagram
    p = np.polyfit(msx, msy, 3)
    q = np.polyfit(msX, msY, 3)
    t = np.linspace(-0.3, 0.8, len(colourf))
    MWfilt=(Gf<p[0]*colourf**3+p[1]*colourf**2+p[2]*colourf+p[3])&(Gf>q[0]*colourf**3+q[1]*colourf**2+q[2]*colourf+q[3])
    
    colourff=colourf[MWfilt]
    lff=lf[MWfilt]
    bff=bf[MWfilt]
    pff=pf[MWfilt]
    mulff=mulf[MWfilt]
    mubff=mubf[MWfilt]
    Gff=Gf[MWfilt]

#    #plotting HR-diagram and limits of main sequence
#    plt.figure(figsize=(16,16))
#    plt.hist2d(colourf,Gf,1000,cmin=1,cmap='jet')
#    plt.xlim(-0.3,0.82)
#    plt.ylim(9,24)
#    plt.gca().invert_yaxis()
#    plt.colorbar(label='Number of stars/bin')
#    plt.xlabel('BP-RP colour')
#    plt.ylabel(r'$M_G$')
#    plt.title(r'HR diagram for filtered sample')
#    plt.tight_layout()
#    plt.plot(t, p[0]*t**3+p[1]*t**2+p[2]*t+p[3], 'r-',linewidth=5)
#    plt.plot(t, q[0]*t**3+q[1]*t**2+q[2]*t+q[3], 'k-',linewidth=5)

    Nstars=len(lff)
    print(Nstars) 
    
    #Defining vectors for calculation of velocities
    ubar=np.array([np.cos(bff)*np.cos(lff),np.cos(bff)*np.sin(lff),np.sin(bff)]).T
    lbar=np.array([-np.sin(lff),np.cos(lff),np.zeros(len(lff))])
    bbar=np.array([-np.sin(bff)*np.cos(lff),-np.sin(bff)*np.sin(lff),np.cos(bff)])
    
    #Calculating mean tangential velocity
    tau=lbar*K*mulff/pff + bbar*K*mubff/pff
    taumean = np.zeros(3)
    for i in range(3):
        taumean[i]=sum(tau[i])/len(tau[0])

    #Calculating Tangential projection matrix, np.einsum takes the vector, 
    #multiplies with each element in the other matrix
    T = np.identity(3) - np.einsum('ij...,i...->ij...',ubar,ubar)
    Tmean = np.sum(T,0)/len(T)
    Tmeaninv=np.linalg.inv(Tmean)

    #Mean velocity
    Vmean=np.dot(Tmeaninv,taumean)

    #Tangential peculiar velocity
    deltau = tau.T - np.dot(T,Vmean)

    #Steps to calculating the dispersion matrix D, eq 8 in the labmanual gives B,
    #using eq. 13, einsum gives Tkm dot Tln as 4D tensor,
    #taking the mean means we can use it.
    B = np.einsum('ij...,i...->ij...',deltau,deltau).mean(0)
    TT = np.einsum('ikm,iln->ikmln',T,T).mean(0)
    D=np.linalg.tensorsolve(TT,B,axes=(0,2))

    #Things for output file
    MeanC=round(np.mean(colourff),3)
    
    MeanAge=np.array([7.94e+07,1.58e+08,2.51e+08,3.16e+08,3.98e+08,5.01e+08,6.31e+08,7.94e+08,7.94e+08,1e+09,1e+09,1.26e+09,1.26e+09,1.58e+09,1.58e+09,2e+09,2.51e+09,3.16e+09,5.01e+09,6.31e+09,1e+10])/2

    Dvec=np.array([D[0,0],D[0,1],D[0,2],D[1,1],D[1,2],D[2,2]])

    with open('P2Data.csv', 'a', newline='') as file:
        writer=csv.writer(file)
        writer.writerow(["%s" % cmin, "%s" % cmax,"%s" % MeanC,"{:.2e}".format(MeanAge[index]),"%s" % round(min(Gff),2),"%s" % round(min(pff),2),"%s" % Nstars, "%s" % round(Vmean[0],2), "%s" % round(Vmean[1],2), "%s" % round(Vmean[2],2), "%s" % round(Dvec[0],2), "%s" % round(Dvec[1],2), "%s" % round(Dvec[2],2), "%s" % round(Dvec[3],2), "%s" % round(Dvec[4],2), "%s" % round(Dvec[5],2)])
