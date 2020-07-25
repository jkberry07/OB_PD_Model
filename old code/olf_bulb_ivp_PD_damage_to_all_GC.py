# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:38:32 2018

@author: wmmjk
"""


#This one does the damage to each of the granule cells (columns of H0)



#Olfactory Bulb Model a la Li/Hopfield and Li/Hertz
#Translated from Avinash's Matlab code
#This one I have the initial condition in odeint as the equilibrium at rest plus noise

#Change Log

#01/28/19   - added a variable "yout" so I could look at both internal state and 
#            output after solving ode
#
#01/29/19                                                             

import numpy as np
import scipy.linalg as lin
from scipy.integrate import solve_ivp
from cellout import cellout,celldiff
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy.signal as signal
from numpy.random import rand
from olf_diffeq import diffeq
from olf_equi import equi
from olf_bulb_fn import olf_bulb_10

plt.close('all')

#INITIALIZING STUFF

Nmitral = 10 #number of mitral cells
Ngranule = 10 #number of granule cells     pg. 383 of Li/Hop
Ndim = Nmitral+Ngranule #total number of cells
t_inh = 25 ; # time when inhalation starts
t_exh = 205; #time when exhalation starts
finalt = 395; # end time of the cycle

#y = zeros(ndim,1);
                          
Sx = 1.43     #Sx,Sx2,Sy,Sy2 are parameters for the activation functions
Sx2 = 0.143
Sy = 2.86     #These are given in Li/Hopfield pg 382, slightly diff in her thesis
Sy2 = 0.286
th = 1       #threshold for the activation function

tau_exh = 33.3333; #Exhale time constant, pg. 382 of Li/Hop
exh_rate = 1/tau_exh

alpha = .15 #decay rate for the neurons
                                #Li/Hop have it as 1/7 or .142 on pg 383

P_odor0=np.zeros((Nmitral,1)) #odor pattern, no odor
P_odor1 = P_odor0 + .00429   #Odor pattern 1
P_odor2 = 1/70*np.array([.6,.5,.5,.5,.3,.6,.4,.5,.5,.5])
P_odor3 = 4/700*np.array([.7,.8,.5,1.2,.7,1.2,.8,.7,.8,.8])
#control_odor = control_order + .00429

#control_odor = np.zeros((Nmitral,1)) #odor input for adaptation

#controllevel = 1 #1 is full adaptation

H0 = np.zeros((Nmitral,Ngranule)) #weight matrix: to mitral from granule
W0 = np.zeros((Ngranule,Nmitral)) #weights: to granule from mitral

Ib = np.ones((Nmitral,1))*.243 #initial external input to mitral cells
Ic = np.ones((Ngranule,1))*.1 #initial input to granule cells, these values are
                              #given on pg 382 of Li/Hop
                              
#Icdecay = .03 #decay constant for control at exhalation
#yescontrol=0 #yescontrol = 1 for central control

signalflag = 1 # 0 for linear output, 1 for activation function

noise = np.zeros((Ndim,1)) #noise in inputs
noiselevel =  .00143
noisewidth = 7 #noise correlation time, given pg 383 Li/Hop as 9, but 7 in thesis

lastnoise = np.zeros((Ndim,1)) #initial time of last noise pule

Icontrol = np.zeros((Ngranule,1)) #central control signal for adaptation
                                #calculated when you call the derivative function

#Init_in = np.append(Ib,Ic,axis=1)




H0 = np.array([[.3,.9,0,0,0,0,0,0,0,.7],\
               [.9,.4,1,0,0,0,0,0,0,0],\
               [0,.8,.3,.8,0,0,0,0,0,0],\
               [0,0,.7,.5,.9,0,0,0,0,0],\
               [0,0,0,.8,.3,.8,0,0,0,0],\
               [0,0,0,0,.7,.3,.9,0,0,0],\
               [0,0,0,0,0,.7,.4,.9,0,0],\
               [0,0,0,0,0,0,.5,.5,.7,0],\
               [0,0,0,0,0,0,0,.9,.3,.9],\
               [.9,0,0,0,0,0,0,0,.8,.3]])

W0 = np.array([[.3,.7,0,0,0,0,0,0,.5,.3],\
               [.3,.2,.5,0,0,0,0,0,0,.7],\
               [0,.1,.3,.5,0,0,0,0,0,0],\
               [0,.5,.2,.2,.5,0,0,0,0,0],\
               [.5,0,0,.5,.1,.9,0,0,0,0],\
               [0,0,0,0,.3,.3,.5,.4,0,0],\
               [0,0,0,.6,0,.2,.3,.5,0,0],\
               [0,0,0,0,0,0,.5,.3,.5,0],\
               [0,0,0,0,0,.2,0,.2,.3,.7],\
               [.7,0,0,0,0,0,0,.2,.3,.5]])


#H0 = H0 + H0*np.random.rand(np.shape(H0))
#W0 = W0+W0*np.random.rand(np.shape(W0))

M = 10   #average over 10 trials for each level of damage

d1 = np.zeros(20)
d2,d3,d4 = np.zeros(20),np.zeros(20),np.zeros(20)
d1it,d2it,d3it,d4it = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
freq,freqit = np.zeros(20),np.zeros(M)
yout2,Sh2 = np.zeros((finalt,Ndim,20)),np.zeros((finalt,Ndim,20))
P_deni = np.zeros((198,Ndim,20))
psi = np.copy(Sh2[:,:Nmitral,:])
IPR = np.zeros(20)
IPR0 = np.zeros(20)
grow_eigs = np.zeros(20)
IPRavg = np.zeros((finalt,20))
Hdamaged = np.copy(H0)  
    
for i in np.arange(20):  #level of damage, from 5% to 100%
    q = np.random.randint(0,10)
    print(q)
    damage = .05*Hdamaged[:,q]

    Hdamaged[:,q] = Hdamaged[:,q]-damage  #decrease column by 5% each time
    
    for m in np.arange(M):
        yout,y0out,Sh,t,Omean1,Oosci1,Omeanbar1,Ooscibar1,freq0,maxlam = \
        olf_bulb_10(H0,W0,P_odor1)
        yout2[:,:,i],y0out2,Sh2[:,:,i],t2,Omean2,Oosci2,Omeanbar2,\
        Ooscibar2,freq[i],grow_eigs[i] = olf_bulb_10(Hdamaged,W0,P_odor1)
        d1it[m] = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
        d2it[m] = 1 - lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
        d3it[m] = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
        d4it[m] = (Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2)
        
        P_den = np.zeros((198,Ndim))
        for p in np.arange(Nmitral):
            f, P_den[:,p] = signal.periodogram(Sh2[:,p,i])
        P_deni[:,:,i] = np.copy(P_den)
        psi = np.zeros(Nmitral)
        for p in np.arange(Nmitral):
            psi[p] = np.sum(P_den[:,p])
        psi = psi/np.sqrt(np.sum(psi**2))
        IPR = 1/np.sum(psi**4)    
        
        IPRavg[i] = IPRavg[i] + IPR
        
    IPRavg[i] = IPRavg[i]/M
    
    d1[i] = np.average(np.abs(d1it))
    d2[i] = np.average(np.abs(d2it))
    d3[i] = np.average(np.abs(d3it))
    d4[i] = np.average(np.abs(d4it))
#        freq[i] = np.average(freqit)     #I'm not going to avg over the freq
# b/c it's not changing that much anyway, and I want it to be clear where it goes away
    
    
    
    #save all the data
   

#******************************************************************************
    
    





#******************************************************************************

