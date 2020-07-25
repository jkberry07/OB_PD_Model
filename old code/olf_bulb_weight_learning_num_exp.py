# -*- coding: utf-8 -*-
"""
July 8th, 2019 at 17:28

@author: wmmjk
"""

#Olfactory Bulb Model a la Li/Hopfield and Li/Hertz
#Translated from Avinash's Matlab code
#This one I have the initial condition in solveivp as the equilibrium at rest plus noise

#Calculates phase differences, distance measure terms, high and low-passed output,
#power spectrum, IPR

#Change Log


import numpy as np
import scipy.linalg as lin
from scipy.integrate import solve_ivp
from cellout import cellout,celldiff
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
#import scipy.signal as signal
#from numpy.random import rand
from olf_diffeq import diffeq
from olf_diffeq import lrndiffeq_Codor
from olf_diffeq import testdiffeq
from olf_diffeq import targetfn
from olf_equi import equi

#plt.close('all')

#INITIALIZING STUFF

Nmitral = 10 #number of mitral cells
Ngranule = 10 #number of granule cells     pg. 383 of Li/Hop
Ndim = Nmitral+Ngranule #total number of cells
t_inh = 25 ; # time when inhalation starts
t_exh = 205; #time when exhalation starts
finalt = 395; # end time of the cycle
t = np.r_[0:finalt]
teval = np.r_[0:finalt] #evaluation times for normal trial
inity = np.zeros((Ndim,1)); # initial values of the neural internal state :
                          # inity[1:nmitral] is the initial values for the 
                          #mitral cells and inity[nmitral+1:end] is the initial
                          #values for the granule cells. Close to the equilibrium value
resty = np.zeros((Ndim,1)); # equilibrium value of the neural internal state when
                          # there is no odor input
#y = zeros(ndim,1);
tau_exh = 33.3333; #Exhale time constant, pg. 382 of Li/Hop
exh_rate = 1/tau_exh

alpha = .15 #decay rate for the neurons, /ms
                                #Li/Hop have it as 1/7 or .142 on pg 383
                          
                          
#define periods of time
preinh = np.r_[0:t_inh]
inhale = np.r_[t_inh:t_exh]
exhale = np.r_[t_exh:finalt]

#time points for evaluation in learning
final_lt = 10*finalt
teval_0 = np.r_[0:final_lt:.2]                                              
lt = np.copy(teval_0) #teval

eta = 1   #learning rate

                          
Sx = 1.43     #Sx,Sx2,Sy,Sy2 are parameters for the activation functions
Sx2 = 0.143
Sy = 2.86     #These are given in Li/Hopfield pg 382, slightly diff in her thesis
Sy2 = 0.286
th = 1       #threshold for the activation function



P_odor0=np.zeros((Nmitral,1)) #odor pattern, no odor
P_odor1 = P_odor0 + .00429   #Odor pattern 1
P_odor2 = 1/70*np.array([.6,.5,.5,.5,.3,.6,.4,.5,.5,.5])
P_odor3 = 4/700*np.array([.7,.8,.5,1.2,.7,1.2,.8,.7,.8,.8])

P_odor4 = 1/70*np.array([1.2,.1,1,.3,.8,.3,.5,.7,1.1,.8])
P_odor5 = np.zeros(Nmitral) + .00429
P_odor5[4] = np.sum(P_odor1)/2


np.random.seed(29)

#Construct Target functions

testfn = np.zeros(finalt)
testfn[preinh] = .004
testfn[inhale] = .004 +.0017*(inhale-t_inh)*(np.sin(.02*2*np.pi*inhale))**2
testfn[exhale] = .004 + (testfn[t_exh-1]/((np.sin(.02*2*np.pi*(t_exh-1)))**2))*\
                  np.exp(-(exhale-t_exh)/tau_exh)*(np.sin(.02*2*np.pi*exhale))**2

target = np.zeros((final_lt,Ndim))

Am = np.zeros(Nmitral)
Bm,phaseshift,Ag,Cg,Bg = np.copy(Am),np.copy(Am),np.copy(Am),np.copy(Am),np.copy(Am)

for i in np.arange(Nmitral):
#    Am[i] = np.random.normal(.01,.00167)
#    Am[i] = np.random.normal(.7,.1)    
    Am[i]=0
#    Bm[i] = np.random.normal(1,.1)
    Bm[i] = np.random.randint(0,14)/10 + np.random.rand()*.1
    phaseshift[i] = np.random.randint(0,785)/1000
#    Ag[i] = np.random.normal(.07,.0067)
#    Ag[i] = np.random.normal(.8,.1)     #for transforming target
    Ag[i]=0
#    Cg[i] = np.random.normal(.0025,.00033)
    Bg[i] = np.random.normal(.8,.2)
    
#    target[preinh,i] = Am[i]
#    target[inhale,i] = Am[i] + Bm[i]*(inhale-t_inh)*(np.sin(.02*2*np.pi*inhale +\
#                          phaseshift[i]))**2
#    target[exhale,i] = Am[i] + (target[t_exh-1,i]/((np.sin(.02*2*np.pi*(t_exh-1)+\
#                          phaseshift[i]))**2))*\
#                          np.exp(-(exhale-t_exh)/tau_exh)*\
#                          (np.sin(.02*2*np.pi*exhale + phaseshift[i]))**2
#    target[preinh,i+10] = Ag[i]
#    target[inhale,i+10] = Ag[i] + Cg[i]*(inhale-t_inh) + Bg[i]*(np.sin(.02*2*np.pi*inhale +\
#                          phaseshift[i] + np.pi/4))**2
#    target[exhale,i+10] = Ag[i] + (target[t_exh-1,i+10])*\
#                          np.exp(-(exhale-t_exh)/tau_exh)*\
#                          (np.sin(.02*2*np.pi*exhale + phaseshift[i] + np.pi/4))**2

#trying simpler target
    
#    target[:,i] = Bm[i]*np.exp(.02*lt)*(np.sin(.02*2*np.pi*lt +\
#                      phaseshift[i]))**2
#    target[:,i+Nmitral] = Bg[i]*np.exp(.02*lt)*(np.sin(.02*2*np.pi*lt +\
#                  phaseshift[i] - np.pi/4))**2
#    target[:,i] = Bm[i]*(np.sin(.02*2*np.pi*lt +\
#                  phaseshift[i]))**2
#    target[:,i] = Bg[i]*(np.sin(.02*2*np.pi*lt +\
#                  phaseshift[i] + np.pi/4))**2

#During the control tests and for mitral cells, the max is around .3; for 
                          #granule cells, it's about .83
#transform to outputs

for i in np.r_[0:finalt]:
    target[i,:Nmitral] = cellout(target[i,:Nmitral],Sx,Sx2,th)
    target[i,Nmitral:] = cellout(target[i,Nmitral:Ndim],Sy,Sy2,th)



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

etas = np.r_[.1:5.1:.1]

convergeH=np.zeros(np.size(etas))
convergeW = np.zeros(np.size(etas))

maxerror = np.zeros(np.size(etas))

frequencies = np.zeros(np.size(etas))


for k in np.arange(np.size(etas)):
    eta = etas[k]
    
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
    
    #W0 = np.array([[.3,.7,0,0,0,0,0,0,.5,.3],\
    #               [.3,.2,.5,0,0,0,0,0,0,.7],\
    #               [0,.1,.3,.5,0,0,0,0,0,0],\
    #               [0,.5,.2,.2,.5,0,0,0,0,0],\
    #               [.5,0,0,.5,.1,.9,0,0,0,0],\
    #               [0,0,0,0,.3,.3,.5,.4,0,0],\
    #               [0,0,0,.6,0,.2,.3,.5,0,0],\
    #               [0,0,0,0,0,0,.5,.3,.5,0],\
    #               [0,0,0,0,0,.2,0,.2,.3,.7],\
    #               [.7,0,0,0,0,0,0,.2,.3,.5]])
    
    W0 = np.array([[.3,.7,0,0,0,0,0,0,0,.3],\
                   [.3,.2,.5,0,0,0,0,0,0,0],\
                   [0,.1,.3,.5,0,0,0,0,0,0],\
                   [0,0,.2,.2,.5,0,0,0,0,0],\
                   [0,0,0,.5,.1,.9,0,0,0,0],\
                   [0,0,0,0,.3,.3,.5,0,0,0],\
                   [0,0,0,0,0,.2,.3,.5,0,0],\
                   [0,0,0,0,0,0,.5,.3,.5,0],\
                   [0,0,0,0,0,0,0,.2,.3,.7],\
                   [.7,0,0,0,0,0,0,0,.3,.5]])
        
    #transform such that e^(sig*H0ij) returns the original H0ij (same for W0)    
    
    H0weights = np.copy(H0[H0>0])
    W0weights = np.copy(W0[W0>0])
    weights = np.append(H0weights,W0[W0>0])
    nzH0 = np.size(H0[H0>0])
    nzW0 = np.size(W0[W0>0])
    
    z = np.zeros(np.size(weights))
    
    
    H0_0 = np.copy(H0)
    W0_0 = np.copy(W0)
    
    locsH0 = np.transpose(np.where(H0>0))
    locsW0 = np.transpose(np.where(W0>0))
    
    #H0 = H0 + H0*np.random.rand(np.shape(H0))
    #W0 = W0+W0*np.random.rand(np.shape(W0))
    
    #H0[:,4] = 0
    
    
    #LEARN THE MATRICES
    sig = .1
    #set initial conditions for all of the variables
    init0 = np.zeros(Ndim+2*nzH0+2*nzW0)
    init0[:Nmitral] = Am        #initial conditions for mitral cells
    init0[Nmitral:Ndim] = Ag    #initial conditions for granule cells
    init0[Ndim:Ndim+nzH0+nzW0] = 0 #initial conditions for dz/dt and du/dt (sensitivities)
    init0[Ndim+nzH0+nzW0:Ndim+2*nzH0+nzW0] = np.log(np.copy(H0[H0>0]))/sig #initial
                                # conditions for H0, will be transformed back
    init0[Ndim+2*nzH0+nzW0:Ndim+2*nzH0+2*nzW0] = np.log(np.copy(W0[W0>0]))/sig  #initial
                                #conditions for W0, will be transformed back via e^(sig*y)
    
    #solve the differential equation
    
    #
    #
    #sol = solve_ivp(lambda t,y: lrndiffeq_all(t,y,target,Nmitral,Ngranule,Ndim,\
    #            t_inh,t_exh,exh_rate,alpha,Sy,Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib,\
    #            teval,Am,Bm,phaseshift,Ag,Cg,Bg),\
    #            [0,395],init0,t_eval = teval,method = 'RK45') 
    
    #try training it to show growing oscillations given a constant odor input
    
    sol = solve_ivp(lambda t,y: lrndiffeq_Codor(t,y,target,Nmitral,Ngranule,Ndim,\
                t_inh,t_exh,exh_rate,alpha,Sy,Sy2,Sx,Sx2,th,H0,W0,locsH0,locsW0, \
                P_odor1,Ic,Ib,teval_0,Am,Bm,phaseshift,Ag,Cg,Bg,eta),\
                [0,final_lt],init0,t_eval = teval_0,method = 'RK45') 
    
    
        
    t = sol.t
    yslearned = sol.y
    ylearned = yslearned[:Ndim,:]
    ylearned = np.transpose(ylearned)
    youtlearned = np.copy(ylearned)
    for i in np.arange(np.size(lt)):
        youtlearned[i,:Nmitral] = cellout(ylearned[i,:Nmitral],Sx,Sx2,th)
        youtlearned[i,Nmitral:] = cellout(ylearned[i,Nmitral:],Sy,Sy2,th)
    
    H0[H0>0] = np.exp(sig*yslearned[Ndim+nzH0+nzW0:Ndim+2*nzH0+nzW0,-1])
    W0[W0>0] = np.exp(sig*yslearned[Ndim+2*nzH0+nzW0:Ndim+2*nzH0+2*nzW0,-1])
    
    target2 = np.zeros((np.size(lt),Ndim))
    for i in np.arange(np.size(lt)):
        target2[i,:] = targetfn(lt[i],Nmitral,Ngranule,Am,Bm,Ag,Bg,phaseshift,Sx,Sx2,Sy,Sy2,th)
    
    
    #Check for convergence
    
    diffH0 = H0-H0_0 #overall difference
    diffW0 = W0-W0_0
    
    H0_change = np.max(yslearned[Ndim+nzH0+nzW0:Ndim+2*nzH0+nzW0,-5:] - \
                       yslearned[Ndim+nzH0+nzW0:Ndim+2*nzH0+nzW0,-6:-1])
    W0_change = np.max(yslearned[Ndim+2*nzH0+nzW0:Ndim+2*nzH0+2*nzW0,-5:] - \
                       yslearned[Ndim+2*nzH0+nzW0:Ndim+2*nzH0+2*nzW0,-6:-1])
    
    convergeH[k] = np.max(H0_change)
    convergeW[k] = np.max(W0_change)
    
    
    
    #*********************************************************
        
        #Solve normal equations using learned weight matrices
    rest0 = np.zeros((Ndim,1))   
    restequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                     t_inh,H0,W0,P_odor0,Ib,Ic),rest0) #about 20 ms to run this
    
    np.random.seed(seed=23)
    init0 = restequi+np.random.rand(Ndim)*.00143 #initial conditions plus some noise
                                               #for no odor input
    np.random.seed()
    
    #teval = teval/5     #For running it five sniff cycles
    
    
    
    sol = solve_ivp(lambda t,y: testdiffeq(lt,y,Nmitral,Ngranule,Ndim,lastnoise,\
                    noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                    Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib),\
                    [0,final_lt],init0,t_eval = teval_0,method = 'RK45') 
    
    ytest = sol.y
    ytest = np.transpose(ytest)
    youttest = np.copy(ytest)
    for i in np.arange(np.size(lt)):
        youttest[i,:Nmitral] = cellout(ytest[i,:Nmitral],Sx,Sx2,th)
        youttest[i,Nmitral:] = cellout(ytest[i,Nmitral:],Sy,Sy2,th)
    
    
    
    
    maxerror[k] = np.max(abs(youttest-target2))/np.max(target2)
    
    
    
#Do a normal sniff-cycle trial with learned weight matrices    
    
    sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
                    noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                    Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib),\
                    [0,395],init0,t_eval = teval,method = 'RK45') 
    
    
    #Try solving with constant odor input (essentially no sniff)
    #sol = solve_ivp(lambda t,y: testdiffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
    #                noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
    #                Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib),\
    #                [0,395],init0,t_eval = teval,method = 'RK45') 
    
    t = sol.t
    y = sol.y
    y = np.transpose(y)
    yout = np.copy(y)
    
    #convert signal into output signal given by the activation fn
    if signalflag ==1:
        for i in np.arange(np.size(t)):
            yout[i,:Nmitral] = cellout(y[i,:Nmitral],Sx,Sx2,th)
            yout[i,Nmitral:] = cellout(y[i,Nmitral:],Sy,Sy2,th)
            
    
    
    
    
    #
    #
    #
    ##******************************************************************************
    #
    ##CALCULATE FIXED POINTS
    #
    ##Calculating equilibrium value with no input
    #rest0 = np.zeros((Ndim,1))
    #
    #restequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
    #                                 t_inh,H0,W0,P_odor0,Ib,Ic),rest0) #about 20 ms to run this
    #
    #np.random.seed(seed=23)
    #init0 = restequi+np.random.rand(Ndim)*.00143 #initial conditions plus some noise
    #                                           #for no odor input
    #np.random.seed()
    ##Now calculate equilibrium value with odor input
    #                                            
    #
    #lastnoise = lastnoise + t_inh - noisewidth  #initialize lastnoise value
    #                                            #But what is it for? to have some
    #                                            #kind of correlation in the noise
                                                
    #find eigenvalues of A to see if input produces oscillating signal
    
    x0 = np.zeros((Ndim,1))
    xequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                     t_inh,H0,W0,P_odor1,Ib,Ic),rest0) 
                        #equilibrium values with some input, about 20 ms to run
    
    
    #
    #
    #
    #
    ##******************************************************************************
    #
    #CALCULATE A AND DETERMINE EXISTENCE OF OSCILLATIONS
    
    diffgy = celldiff(xequi[Nmitral:],Sy,Sy2,th)
    diffgx = celldiff(xequi[0:Nmitral],Sx,Sx2,th)
    
    H1 = np.dot(H0,diffgy)  
    W1 = np.dot(W0,diffgx)  #intermediate step in constructing A
    
    A = np.dot(H1,W1)   #Construct A
    
    dA,vA = lin.eig(A) #about 20 ms to run this
                        #Find eigenvalues of A
    
    diff = (1j)*(dA)**.5 - alpha   #criteria for a growing oscillation
    
    negsum = -(1j)*(dA)**.5 - alpha #Same
    
    diff_re = np.real(diff)     
                               #Take the real part
    negsum_re = np.real(negsum)
    
    #do an argmax to return the eigenvalue that will cause the fastest growing oscillations
    #Then do a spectrograph to track the growth of the associated freq through time
    
    indices = np.where(diff_re>0)   #Find the indices where the criteria is met
    indices2 = np.where(negsum_re>0)
    
    #eigenvalues that could lead to growing oscillations
    candidates = [np.real((dA[indices])**.5),np.real((dA[indices2])**.5)]
    largest = np.argmax(diff_re)
    
    check = np.size(indices)
    check2 = np.size(indices2)
    
    
    if check==0 and check2==0:
       # print("No Odor Recognized")
        frequencies[k] = 0
    else:
    #    dominant_freq = np.max(candidates)/(2*np.pi) #find frequency of the dominant mode
                                                #Divide by 2pi to get to cycles/ms
        dominant_freq = np.real((dA[largest])**.5)/(2*np.pi)
        frequencies[k] = dominant_freq
#        print("Odor detected. Eigenvalues:",dA[indices],dA[indices2],\
#              "\nEigenvectors:",vA[indices],vA[indices2],\
#              "\nDominant Frequency:",dominant_freq)
    
    
#
#
#
#
##*************************************************************************
#
##SOLVE DIFFERENTIAL EQUATIONS TO GET INPUT AND OUTPUTS AS FN'S OF t
#  
#    #differential equation to solve
#teval = np.r_[0:finalt]                                              
#
##solve the differential equation
#sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
#                noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
#                Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib),\
#                [0,395],init0,t_eval = teval,method = 'RK45') 
#t = sol.t
#y = sol.y
#y = np.transpose(y)
#yout = np.copy(y)
#
##convert signal into output signal given by the activation fn
#if signalflag ==1:
#    for i in np.arange(np.size(t)):
#        yout[i,:Nmitral] = cellout(y[i,:Nmitral],Sx,Sx2,th)
#        yout[i,Nmitral:] = cellout(y[i,Nmitral:],Sy,Sy2,th)
#        
##solve diffeq for P_odor = 0
##first, reinitialize lastnoise & noise
#noise = np.zeros((Ndim,1))
#lastnoise = np.zeros((Ndim,1))
#lastnoise = lastnoise + t_inh - noisewidth
#
#sol0 = sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
#                noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
#                Sy2,Sx,Sx2,th,H0,W0,P_odor0,Ic,Ib),\
#                [0,395],init0,t_eval = teval,method = 'RK45')
#y0 = sol0.y
#t0 = sol0.t
#y0 = np.transpose(y0)
#y0out = np.copy(y0)
#
##convert signal into output signal given by the activation fn
#if signalflag ==1:
#    for i in np.arange(np.size(t)):
#        y0out[i,:Nmitral] = cellout(y0[i,:Nmitral],Sx,Sx2,th)
#        y0out[i,Nmitral:] = cellout(y0[i,Nmitral:],Sy,Sy2,th)
#
#
#
#
#
#
#
##*****************************************************************************
#
##SIGNAL PROCESSING
#
##Filtering the signal - O_mean: Lowpass fitered signal, under 20 Hz
##S_h: Highpass filtered signal, over 20 Hz
#
#fs = 1/(.001*(t[1]-t[0]))  #sampling freq, converting from ms to sec
#
#f_c = 20/fs     # Cutoff freq at 20 Hz, written as a ratio of fc to sample freq
#
#flter = np.sinc(2*f_c*(t - (finalt-1)/2))*np.blackman(finalt) #creating the
#                                                    #windowed sinc filter
#                                                    #centered at the middle
#                                                    #of the time data
#flter = flter/np.sum(flter)  #normalize
#
#hpflter = -np.copy(flter)
#hpflter[int((finalt-1)/2)] += 1  #convert the LP filter into a HP filter
#
#Sh = np.zeros(np.shape(yout))
#Sl = np.copy(Sh)
#Sl0 = np.copy(Sh)
#Sbp = np.copy(Sh)
#
#for i in np.arange(Ndim):
#    Sh[:,i] = np.convolve(yout[:,i],hpflter,mode='same')
#    Sl[:,i] = np.convolve(yout[:,i],flter,mode='same')
#    Sl0[:,i] = np.convolve(y0out[:,i],flter,mode='same')
#    
##find the oscillation period Tosc (Tosc must be greater than 5 ms to exclude noise)
#Tosc0 = np.zeros(np.size(np.arange(5,50)))
#for i in np.arange(5,50):
#    Sh_shifted=np.roll(Sh,i,axis=0)
#    Tosc0[i-5] = np.sum(np.diagonal(np.dot(np.transpose(Sh[:,:Nmitral]),Sh_shifted[:,:Nmitral])))
#    #That is, do the correlation matrix (time correlation), take the diagonal to
#    #get the autocorrelations, and find the max
#Tosc = np.argmax(Tosc0)
#Tosc = Tosc + 5
#
#f_c2 = 1000*(1.3/Tosc)/fs  #Filter out components with frequencies higher than this
#                        #to get rid of noise effects in cross-correlation
#                        #times 1000 to get units right
#
#flter2 = np.sinc(2*f_c2*(t - (finalt-1)/2))*np.blackman(finalt)
#flter2 = flter2/np.sum(flter2)
#
#for i in np.arange(Ndim):
#    Sbp[:,i] = np.convolve(Sh[:,i],flter2,mode='same')
#
#
#
##CALCULATE THE DISTANCE MEASURES
#
##calculate phase via cross-correlation with each cell
#phase = np.zeros(Nmitral)
#
#for i in np.arange(1,Nmitral):
#    crosscor = signal.correlate(Sbp[:,0],Sbp[:,i])
#    tdiff = np.argmax(crosscor)-(finalt-1)
#    phase[i] = tdiff/Tosc * 2*np.pi
#    
##Problem with the method below is that it will only give values from 0 to pi
##for i in np.arange(1,Nmitral):
##    phase[i]=np.arccos(np.dot(Sbp[:,0],Sbp[:,i])/(lin.norm(Sbp[:,0])*lin.norm(Sbp[:,i])))
#
#OsciAmp = np.zeros(Nmitral)
#Oosci = np.copy(OsciAmp)*0j
#Omean = np.zeros(Nmitral)
#
#for i in np.arange(Nmitral):
#    OsciAmp[i] = np.sqrt(np.sum(Sh[:,i]**2)/np.size(Sh[:,i]))
#    Oosci[i] = OsciAmp[i]*np.exp(1j*phase[i])
#    Omean[i] = np.average(Sl[:,i] - Sl0[:,i])
#
#Omean = np.maximum(Omean,0)
#
#Ooscibar = np.sqrt(np.dot(Oosci,np.conjugate(Oosci)))/Nmitral #can't just square b/c it's complex
#Omeanbar = np.sqrt(np.dot(Omean,Omean))/Nmitral
#
##d1 = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
##d2 = 1 - lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
##
##d3 = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
##d4 = (Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2)
#
#
##Calculate IPR
#P_den = np.zeros((198,Ndim))
#for i in np.arange(Nmitral):
#    f, P_den[:,i] = signal.periodogram(Sh[:,i])  #periodogram returns a list of
#                                                 #frequencies and the power density
#psi = np.zeros(Nmitral)
#for i in np.arange(Nmitral):
#    psi[i] = np.sum(P_den[:,i])
#psi = psi/np.sqrt(np.sum(psi**2))
#IPR0 = 1/np.sum(psi**4)    
#
#
#
#
##******************************************************************************
#
##PLOT STUFF
# 
fig1 = plt.figure(10,figsize = (16,8.5)) #figsize (26,12) is full screen
fig1.canvas.set_window_title('cell activity')  #cell outputs
fig1.suptitle('Cell Activity, output vs. time')
for i in np.arange(Nmitral):
    plt.subplot(5,2,i+1)
    plt.plot(t,yout[:,i])
    #plt.ylim(np.min(yout[:,Nmitral:]),np.max(yout[:,Nmitral:]))
for i in np.arange(Ngranule):
    plt.subplot(5,2,i+1)
    plt.plot(t,yout[:,i+Nmitral])
#    plt.plot(t,target[:,i])
#    plt.plot(t,target[:,i+Nmitral])
      
fig2 = plt.figure(1,figsize = (16,8.5)) #figsize (26,12) is full screen
fig2.canvas.set_window_title('cell activity - learning')  #cell outputs
fig2.suptitle('learning-dancers, output vs. time')
for i in np.arange(Nmitral):
    plt.subplot(5,2,i+1)
    plt.plot(lt,youttest[:,i])
    #plt.ylim(np.min(youtlearned[:,Nmitral:]),np.max(youtlearned[:,Nmitral:]))
for i in np.arange(Ngranule):
    plt.subplot(5,2,i+1)
    plt.plot(lt,youttest[:,i+Nmitral])
 
target_error= youttest-target2    
fig3 = plt.figure(2,figsize = (16,8.5)) #figsize (26,12) is full screen
fig3.canvas.set_window_title('cell activity minus target')  #cell outputs
fig3.suptitle('error, output vs. time')
for i in np.arange(Nmitral):
    plt.subplot(5,2,i+1)
    plt.plot(lt,target_error[:,i])
    #plt.ylim(np.min(youtlearned[:,Nmitral:]),np.max(youtlearned[:,Nmitral:]))
for i in np.arange(Ngranule):
    plt.subplot(5,2,i+1)
    plt.plot(lt,target_error[:,i+Nmitral])  
    
fig4 = plt.figure(3,figsize = (16,8.5)) #figsize (26,12) is full screen
fig4.canvas.set_window_title('target')  #cell outputs
fig4.suptitle('target, output vs. time')
for i in np.arange(Nmitral):
    plt.subplot(5,2,i+1)
    plt.plot(lt,target2[:,i])
    #plt.ylim(np.min(youtlearned[:,Nmitral:]),np.max(youtlearned[:,Nmitral:]))
for i in np.arange(Ngranule):
    plt.subplot(5,2,i+1)
    plt.plot(lt,target2[:,i+Nmitral])  

##fig2 = plt.figure(2,figsize = (16,10))  #Plot highpass filtered outputs for each cell
##fig2.suptitle('Cell Activity - Highpass Filtered, output vs. time')
##for i in np.arange(Nmitral):
##    plt.subplot(5,2,i+1)
##    plt.plot(t,Sh[:,i])
##    plt.ylim(np.min(Sh[:,:Nmitral]),np.max(Sh[:,:Nmitral]))
###for i in np.arange(Ngranule):
###    plt.subplot(5,2,i+1)
###    plt.plot(t,Sh[:,i+Nmitral])
##    
###fig3 = plt.figure(3,figsize = (16,10))  #Plot the fourier transform of each output
###fig3.suptitle('Fourier Transform, signal vs. frequency')
###Shfreq = np.fft.rfftfreq(np.size(Sh[:,0]))  #get freq points for FT plots
###for i in np.arange(Nmitral):
###    plt.subplot(5,2,i+1)
###    plt.plot(Shfreq,np.fft.rfft(Sh[:,i]))  #plot the Fourier Transform of each
###    plt.xlim(0,.2)
###for i in np.arange(Ngranule):               #cell's activity
###    plt.subplot(5,2,i+1)
###    plt.plot(Shfreq,np.fft.rfft(Sh[:,i+Nmitral]))
###    plt.xlim(0,.2)
###plt.xlabel('Frequency (kHz) Theoretical Dominant Freq = ' + str(dominant_freq))
##
##fig4 = plt.figure(4, figsize = (16,10))  #Plot the periodogram for each output
##fig4.suptitle('Periodogram, power density vs. frequency')
##for i in np.arange(Nmitral):                   
##    plt.subplot(5,2,i+1)                    
##    plt.plot(f,P_den[:,i])
##    plt.xlim(0,.2)
##    plt.ylim(0,np.max(P_den[:,:Nmitral]))
##for i in np.arange(Ngranule):
##    f, P_den[:,i+Nmitral] = signal.periodogram(Sh[:,i+Nmitral])
##    plt.subplot(5,2,i+1)
##    plt.plot(f,P_den[:,i+Nmitral])
###    marker[17] = np.max(P_den[:,i+Nmitral])
###    plt.plot(f,marker)
##    plt.xlim(0,.2)
#    
#
##plt.figure(5)  #Plot the spectrogram for each output
##for i in np.arange(Nmitral):
##    f2,t2,Spect = signal.spectrogram(Sh[:,i], nperseg = 30)   #Spectrogram returns a list of
##    plt.subplot(5,2,i+1)                        #frequencies, times, and a 2D
##    plt.pcolormesh(t2,f2,Spect)              #array of values for each f and t
##    plt.ylim(0,.2)
##plt.ylabel('Frequency (kHz)')
##plt.xlabel('time (ms)')
#    
#

