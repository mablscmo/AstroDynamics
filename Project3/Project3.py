# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:15:19 2019

@author: monte
"""

import matplotlib.pyplot as plt 
import numpy as np
import numpy.random
from scipy.integrate import ode

font = {'weight' : 'bold'}
plt.rcParams.update({'font.size': 25})

#Gravitational constant in pc (km/s)^2 M_\odot^-1
G = 0.004301

#Loads data and creates a list of the total velocities. The sorted list is used to find interesting stars
Data = np.loadtxt('galvel.dat')
vlist = np.sqrt(Data[:,1]**2+Data[:,2]**2+Data[:,3]**2)
vsort = np.sort(vlist)
#Similar lists for galactocentric velocities
vlist2 = np.sqrt((Data[:,1]-23.67)**2+(Data[:,2]+227.67)**2+(Data[:,3]-8.83)**2)
vsort2 = np.sort(vlist2)

#Test for circular orbit
params = np.array([-8000,0,0,0,220,0])
#Solar parameters
params = np.array([-8000,0,0,-23.67,227.67,-8.83])

Name = 'the Sun'


# ##Stars investigated by the code. params adjusted to reflect velocities being relative to the Sun
# #122 38-Cas  525 del-Lep  1083 gam-Cen  1481 36-Oph  1936 ν-Ind  2022 gam-Psc 
# index = 2022
# location = np.array([-8000,0,0])
# params = np.append(location,Data[:,1:][index] + np.array([-23.67,227.67,-8.83]))
# print(params)

# #Looks up the name of the star in the Hipporcos catalogue
# Name = 'HIP %s' % int(Data[index][0])

#Constants for the galactic potential ab bb Mb ad bd Md rc Mc
Pot = np.array([0, 277, 1.12*10**10, 3700, 200, 8.07*10**10, 6000, 5.0*10**10])

def orbit(t, params, Pot=Pot):
    '''Takes six parameters in a array-like params argument and returns their derivatives.
    Also takes a t argument which isen't used in the function to make it compatible with 
    the scipy ode integrator used later. The Pot argument sets the shape of the Milky Way
    potential the stars are integrated in.
    The velocities are returned as the derivatives of the positions while the derivatives 
    of the different potential components are calculated to compute the acceleration.'''
    #Unpacks arguments
    x,y,z,u,v,w = params
    ab, bb, Mb, ad, bd, Md, rc, Mc = Pot
    #Creates R and r variables for the potentials
    R = np.sqrt(x**2+y**2)
    r = np.sqrt(x**2+y**2+z**2)
    
    #Miyamoto-Nagai potential derivatives for the bulge. A x/y/z term is left out to enable reuse.
    MNBxy = G*Mb/((R**2 + (ab+np.sqrt(z**2+bb**2))**2)**(3/2))
    MNBz = G*Mb*(ab+np.sqrt(bb**2+z**2))/(np.sqrt(bb**2+z**2)*(R**2+(ab+np.sqrt(bb**2+z**2))**2))**(3/2)
    
    #Miyamoto-Nagai potential derivatives for the disc.
    MNDxy = G*Md/((R**2 + (ad+np.sqrt(z**2+bd**2))**2)**(3/2))
    MNDz = G*Md*(ad+np.sqrt(bd**2+z**2))/(np.sqrt(bd**2+z**2)*(R**2+(ad+np.sqrt(bd**2+z**2))**2))**(3/2)
    
    #Derivative of the halo potential.
    Halo = (G*Mc/rc)*((1/(r**2 * ((r**2/rc**2)+1)) + 1/(rc**2 * ((r**2/rc**2)+1)) - rc*np.arctan(r/rc)/r**3))
    
    #Combines the derivatives of the potentials to compute the accelerations.
    ax = -x*(MNBxy+MNDxy+Halo)
    ay = -y*(MNBxy+MNDxy+Halo)
    az = -z*(MNBz+MNDz+Halo)
    return [u,v,w,ax,ay,az]


#Starts the use of the function, setting the starting time to zero. Sets a fixed timestep, 
#can be adjusted to speed up computation or return more precise simulations.
#Set dt and tmax to be negative and change < to > in the while loop to look backwards in time.
    
t0 = 0
dt = 0.3

#Creates an ode object on which to work, and sets the initial values in accordance to the ones 
#imported from Hipparcos above.
r = ode(orbit).set_integrator('dopri5', atol=1e-05)
r.set_initial_value(params, t0)


#Different maximum timespans, computed from the 0.9778 Myr unit caused by the choice of units in G.
#250 Myr
# tmax = 256
#1 Gyr
tmax = 1023
#1.5 gyr
#tmax = 1534
# 2 Gyr
# tmax = 2045
#4 Gyr
# tmax = 4092
# #10 Gyr
# tmax = 10230


#While loop takes a set number of steps until the maximum time has been reached. Transforms into 
#numpy arrays after to make eventual manipulation easier.
coords = []
time= []
while r.t < tmax:
    coords.append(r.integrate(r.t+dt))
    time.append(r.t)
coords = np.asarray(coords)
time = np.asarray(time)

#Unpacks position and velocity trajectories from the integration.
x, y, z, u, v, w = coords[:,0], coords[:,1], coords[:,2], coords[:,3], coords[:,4], coords[:,5]
R = np.sqrt(x**2+y**2)
r = np.sqrt(x**2+y**2+z**2)
vavg = np.sqrt(u**2+v**2+w**2)
zabs = np.abs(z)

#Projected future stellar orbit
plt.figure(figsize=(12,12))
plt.plot(x,y,'sienna',linewidth=5)
plt.title('Orbit of {n} for {yr} Myr'.format(n=Name, yr=round(tmax*0.9778)))
plt.xlabel('x [pc]')
plt.ylabel('y [pc]')
plt.xlim((-25000,25000))
plt.ylim((-25000,25000))
plt.tight_layout()

#Orbital period
period = [0]
for i in range(len(time)-1):
    if np.sign(y[i]) != np.sign(y[i+1]):
        period.append(time[i])
period = np.asarray(period)

MeanOrbit = np.mean(np.diff(period))*2*0.9778

print('Mean orbital period of {n} is {MOP} Myr'.format(n = Name, MOP = round(MeanOrbit)))

#Max and min distance
rmax = np.amax(r)
rmin = np.amin(r)
print('rmax = {rx} pc, rmin = {rm} pc'.format(rx=round(rmax), rm=round(rmin)))


#3D stellar orbit, limits manually set to prevent distortion.
fig = plt.figure(figsize=(14,14))
ax = plt.axes(projection='3d')
p = ax.scatter3D(x,y,z,c=time,cmap='copper')

ax.set_xlim3d(-10000,10000)
ax.set_ylim3d(-10000,10000)
ax.set_zlim3d(-10000,10000)

ax.set_xticks([-8000, -4000, 0, 4000, 8000])
ax.set_yticks([-8000, -4000, 0, 4000, 8000])
ax.set_zticks([-8000, -4000, 0, 4000, 8000])

ax.yaxis._axinfo['label']['space_factor'] = 3.0

ax.set_title('3D orbit of {n} for {yr} Myr'.format(n=Name, yr=round(tmax*0.9778)))
ax.set_xlabel('x [pc]',rotation=0)
ax.set_ylabel('y [pc]',rotation=0)
ax.set_zlabel('z [pc]')
fig.colorbar(p,label='Time [Myr]')



#Evolution of radial oscillation
plt.figure(figsize=(12,12))
plt.plot(time/0.9778,R,'sienna',linewidth=5)
plt.title('Radial oscillation of {n} for {yr} Myr'.format(n=Name, yr=round(tmax*0.9778)))
plt.xlabel('Time [Myr]')
plt.ylabel('Radial distance to GC [pc]')
plt.ylim((1800,18300))
plt.tight_layout()

#Radial oscillation period
Rperiod = []
Rosc = R-np.mean(R)
for i in range(len(time)-1):
    if np.sign(Rosc[i]) != np.sign(Rosc[i+1]):
        Rperiod.append(time[i])
Rperiod = np.asarray(Rperiod)

MeanROsc = np.mean(np.diff(Rperiod))*2*0.9778

print('Mean radial oscillation period of {n} is {MOP} Myr'.format(n = Name, MOP = round(MeanROsc)))


#Evolution of vertical oscillation
plt.figure(figsize=(12,12))
plt.plot(time/0.9778,z,'sienna',linewidth=5)
plt.title('Vertical oscillation of {n} for {yr} Myr'.format(n=Name, yr=round(tmax*0.9778)))
plt.xlabel('Time [Myr]')
plt.ylabel('Height above galactic plane [pc]')
plt.ylim((-6300,6300))
plt.tight_layout()

#Max and min height
Zmax = np.amax(zabs)
print('Zmax = %s pc' % round(Zmax))


#Vertical oscillation period
Zperiod = []
Zosc = z-np.mean(z)
for i in range(len(time)-1):
    if np.sign(Zosc[i]) != np.sign(Zosc[i+1]):
        Zperiod.append(time[i])
Zperiod = np.asarray(Zperiod)

MeanZOsc = np.mean(np.diff(Zperiod))*2*0.9778

print('Mean vertical oscillation period of {n} is {MOP} Myr'.format(n = Name, MOP = round(MeanZOsc)))
