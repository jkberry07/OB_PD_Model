# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:48:10 2019

@author: wmmjk
"""

####Olfactory Bulb - Matrix analysis

#How different is each element from each other?
#How many (out of 100) result in oscillations, as indicated by the eigenvalues
#What is the average weight strength




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

Nmatrices = 1

Nmitral = 20 #number of mitral cells
Ngranule = 20 #number of granule cells     pg. 383 of Li/Hop
Ndim = Nmitral+Ngranule #total number of cells
t_inh = 25 ; # time when inhalation starts
t_exh = 205; #time when exhalation starts
finalt = 395; # end time of the cycle
inity = np.zeros((Ndim,1)); # initial values of the neural internal state :
                          # inity[1:nmitral] is the initial values for the 
                          #mitral cells and inity[nmitral+1:end] is the initial
                          #values for the granule cells. Close to the equilibrium value
resty = np.zeros((Ndim,1)); # equilibrium value of the neural internal state when
                          # there is no odor input
#y = zeros(ndim,1);
                          
Sx = 1.43     #Sx,Sx2,Sy,Sy2 are parameters for the activation functions
Sx2 = 0.143
Sy = 2.86     #These are given in Li/Hopfield pg 382, slightly diff in her thesis
Sy2 = 0.286
th = 1       #threshold for the activation function

tau_exh = 33.3333; #Exhale time constant, pg. 382 of Li/Hop
exh_rate = 1/tau_exh

alpha = .15 #decay rate for the neurons, /ms
                                #Li/Hop have it as 1/7 or .142 on pg 383

P_odor0=np.zeros((Nmitral,1)) #odor pattern, no odor
P_odor1 = P_odor0 + .00429   #Odor pattern 1
P_odor2 = 1/70*np.array([.6,.5,.5,.5,.3,.6,.4,.5,.5,.5])
P_odor3 = 4/700*np.array([.7,.8,.5,1.2,.7,1.2,.8,.7,.8,.8])

P_odor4 = 1/70*np.array([1.2,.1,1,.3,.8,.3,.5,.7,1.1,.8])
P_odor5 = np.zeros(Nmitral) + .00429
P_odor5[4] = np.sum(P_odor1)/2
#control_odor = control_order + .00429

#control_odor = np.zeros((Nmitral,1)) #odor input for adaptation

#controllevel = 1 #1 is full adaptation

H0 = np.zeros((Nmitral,Ngranule)) #weight matrix: to mitral from granule
W0 = np.zeros((Ngranule,Nmitral)) #weights: to granule from mitral

Ib = np.ones((Nmitral,1))*.243 #initial external input to mitral cells
Ic = np.ones((Ngranule,1))*.1 #initial input to granule cells, these values are
                              #given on pg 382 of Li/Hop
     
H0fl = "H0_20"
W0fl = "W0_20"
                   
H0s = np.zeros((Nmatrices,np.shape(H0)[0],np.shape(H0)[1]))
W0s = np.zeros((Nmatrices,np.shape(W0)[0],np.shape(W0)[1]))

for p in np.arange(Nmatrices):
    H0fl = "H0_20" + str(p) + ".npy"
    W0fl = "W0_20" + str(p) + ".npy"
    H0s[p,:,:] = np.load(H0fl)
    W0s[p,:,:] = np.load(W0fl)

H00,W00 = H0s[0,:,:],W0s[0,:,:]
nzH0 = np.size(H00[H00>0])
nzW0 = np.size(W00[W00>0])

freqs = np.load("frequencies.npy")
Nresponses = np.size(freqs[freqs>0])

diffsH0 = np.zeros((1,np.shape(H0)[0],np.shape(H0)[1]))
diffsW0 = np.zeros((1,np.shape(W0)[0],np.shape(W0)[1]))

for n in np.arange(Nmatrices):
    i = 1
    while (n+i)<Nmatrices:
        diffsH0 = np.abs(np.append(diffsH0, H0s[n,:,:] - H0s[n+i,:,:]))
        diffsW0 = np.abs(np.append(diffsW0, W0s[n,:,:] - W0s[n+i,:,:]))
        i+=1

Ndiffs = np.math.factorial(Nmatrices)/(np.math.factorial(2)*np.math.factorial(Nmatrices-2))

avgdiffsH0,avgdiffsW0 = np.zeros(Ndiffs),np.zeros(Ndiffs)
for n in np.arange(Ndiffs):
    avgdiffsH0 = np.sum(diffsH0[n,:,:])
    avgdiffsW0 = np.sum(diffsW0[n,:,:])
    
#plt.figure(0)
#plt.title("average difference between weight entries in H0 matrices")
#plt.hist(avgdiffsH0)
#
#plt.figure(1)
#plt.title("average difference between weight entries in W0 matrices")
#plt.hist(avgdiffsW0)
#
#diffhist_H0 = np.histogram(diffsH0[diffsH0>0])
#plt.figure(2)
#plt.title("difference between corresponding weight entries in H0 matrices")
#plt.hist(diffsH0[diffsH0>0])
#
#plt.figure(3)
#plt.title("difference between corresponding weight entries in W0 matrices")
#plt.hist(diffsH0[diffsW0>0])
#
#hist_H0 = np.histogram(H0s[H0s>0])
#plt.figure(4)
#plt.title("weight strengths in H0 matrices")
#plt.hist(H0s[H0s>0])
#
#plt.figure(5)
#plt.title("weight strengths in W0 matrices")
#plt.hist(W0s[W0s>0])
#
#plt.figure(6)
#plt.hist(freqs)