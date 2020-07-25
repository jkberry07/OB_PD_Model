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

                                                           
import numpy as np
import scipy.linalg as lin
import scipy.stats as stats
import scipy.signal as signal
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import time
import os
import sys
sys.path.append(os.getcwd())

tm1 = time.time()

Nmitral0 = 50  #define network size in parent process



#####Chapter 1 - The Output functions########################

def cellout(x,s1,s2,th):
    g = np.zeros(np.shape(x))
    for i in np.r_[0:np.size(x)]:
        if x[i] < th:
            g[i] = s2 + s2*np.tanh((x[i]-th)/s2)
        else:
            g[i] = s2 + s1*np.tanh((x[i]-th)/s1)
    
    return g

def celldiff(x,s1,s2,th):
    #Returns the differentiated outputs in a diagonal matrix for calculating 
    #the matrix A
    Gdiff = np.zeros((np.size(x),np.size(x)))
    for i in np.r_[0:np.size(x)]:
        if x[i] < th:
            Gdiff[i,i] = (1 - (np.tanh((x[i]-th)/s2))**2)
        else:
            Gdiff[i,i] = (1 - (np.tanh((x[i]-th)/s1))**2)
            
    return Gdiff




######Chapter 2 - The Equations###########################
def equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,t_inh,H0,W0,P_odor,Ib,Ic):
    F = np.zeros(Ndim)
    gx = cellout(x[0:Nmitral],Sx,Sx2,th)  #calculate output from internal state
    gy = cellout(x[Nmitral:],Sy,Sy2,th)   #for mitral and granule cells respectively
    F[0:Nmitral] = np.ravel(-np.dot(H0,gy)) - np.ravel(alpha*x[0:Nmitral]) + np.ravel(Ib) + \
        np.ravel(P_odor*(180-t_inh)) #180 is 25 ms before the end of inhale
    F[Nmitral:] = np.ravel(np.dot(W0,gx)) - np.ravel(alpha*x[Nmitral:]) + np.ravel(Ic)
    return F



def diffeq(t,x,Nmitral,Ngranule,Ndim,lastnoise,noise,noisewidth,noiselevel,\
           t_inh,t_exh,exh_rate,alpha,Sy,Sy2,Sx,Sx2,th,H0,W0,P_odor,Ic,Ib):
    y = x
    dydt = np.zeros(np.shape(y))
    for i in np.r_[0:Nmitral]:
        if t < t_inh:
            dydt[i] = (t-lastnoise[i])*noise[i] - np.dot(np.reshape(H0[i,:],\
                (1,Ngranule)), cellout(y[Nmitral:],Sy,Sy2,th)) - alpha*y[i] + \
                Ib[i]
        elif t < t_exh:
            dydt[i] = (t-lastnoise[i])*noise[i] - np.dot(np.reshape(H0[i,:],\
                (1,Ngranule)), cellout(y[Nmitral:],Sy,Sy2,th)) - alpha*y[i] + \
                Ib[i] + P_odor[i]*(t-t_inh)
        else:
            dydt[i] = (t-lastnoise[i])*noise[i] - np.dot(np.reshape(H0[i,:],\
                (1,Ngranule)), cellout(y[Nmitral:],Sy,Sy2,th)) - alpha*y[i] + \
                Ib[i] + P_odor[i]*(t-t_inh)*np.exp(-exh_rate*(t-t_exh))
    
    for i in np.r_[Nmitral:Ndim]:
        dydt[i] = (t-lastnoise[i])*noise[i] + np.dot(np.reshape(\
            W0[i-Nmitral,:],(1,Nmitral)),cellout(y[:Nmitral],Sx,Sx2,th)) - \
            alpha*y[i] + Ic[i-Nmitral]
            
    for i in np.r_[0:Ndim]:
        if (t-lastnoise[i])/noisewidth > .8 + np.random.rand():
            lastnoise[i] = t
            noise[i] = noiselevel*(2*np.random.rand() -1)  #to get noise btwn
                                                          #-noiselevel and +nslvl
    return dydt



#########Chapter3 - The solver##############################
def olf_bulb_10(Nmitral,H_in,W_in,P_odor_in):   
#    Nmitral = 10 #number of mitral cells
    Ngranule = np.copy(Nmitral) #number of granule cells     pg. 383 of Li/Hop
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

    
    H0 = H_in #weight matrix: to mitral from granule
    W0 = W_in #weights: to granule from mitral
    
    Ib = np.ones((Nmitral,1))*.243 #initial external input to mitral cells
    Ic = np.ones((Ngranule,1))*.1 #initial input to granule cells, these values are
                                  #given on pg 382 of Li/Hop
                                  
    
    signalflag = 1 # 0 for linear output, 1 for activation function
    
    noise = np.zeros((Ndim,1)) #noise in inputs
    noiselevel =  .00143
    noisewidth = 7 #noise correlation time, given pg 383 Li/Hop as 9, but 7 in thesis
    
    lastnoise = np.zeros((Ndim,1)) #initial time of last noise pule
    

    
    #******************************************************************************
    
    #CALCULATE FIXED POINTS
    
    #Calculating equilibrium value with no input
    rest0 = np.zeros((Ndim,1))
    
    restequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                     t_inh,H0,W0,P_odor0,Ib,Ic),rest0) #about 20 ms to run this
    
    np.random.seed(seed=23)
    #init0 = restequi+np.random.rand(Ndim)*.00143 #initial conditions plus some noise
                                               #for no odor input
    init0 = restequi+np.random.rand(Ndim)*.00143 #initial conditions plus some noise
                                           #for no odor input                 
    np.random.seed()
    #Now calculate equilibrium value with odor input
                                                
    
    lastnoise = lastnoise + t_inh - noisewidth  #initialize lastnoise value
                                                #But what is it for? to have some
                                                #kind of correlation in the noise
                                                
    #find eigenvalues of A to see if input produces oscillating signal
    
    xequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                     t_inh,H0,W0,P_odor_in,Ib,Ic),rest0) 
                        #equilibrium values with some input, about 20 ms to run
    
    
    
    
    
    
    #******************************************************************************
    
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
#    candidates = np.append(np.real((dA[indices])**.5),np.real((dA[indices2])**.5))
    largest = np.argmax(diff_re)
    
    check = np.size(indices)
    check2 = np.size(indices2)
    
    
    if check==0 and check2==0:
    #    print("No Odor Recognized")
        dominant_freq = 0
    else:
        dominant_freq = np.real((dA[largest])**.5)/(2*np.pi) #find frequency of the dominant mode
                                                #Divide by 2pi to get to cycles/ms
    #    print("Odor detected. Eigenvalues:",dA[indices],dA[indices2],\
    #          "\nEigenvectors:",vA[indices],vA[indices2],\
    #          "\nDominant Frequency:",dominant_freq)
    
    
    #*************************************************************************
    
    #SOLVE DIFFERENTIAL EQUATIONS TO GET INPUT AND OUTPUTS AS FN'S OF t
      
        #differential equation to solve
    teval = np.r_[0:finalt]                                              
    
    #solve the differential equation
    sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
                    noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                    Sy2,Sx,Sx2,th,H0,W0,P_odor_in,Ic,Ib),\
                    [0,395],init0,t_eval = teval,method = 'RK45') 
    t = sol.t
    y = sol.y
    y = np.transpose(y)
    yout = np.copy(y)
    
    #convert signal into output signal given by the activation fn
    if signalflag ==1:
        for i in np.arange(np.size(t)):
            yout[i,:Nmitral] = cellout(y[i,:Nmitral],Sx,Sx2,th)
            yout[i,Nmitral:] = cellout(y[i,Nmitral:],Sy,Sy2,th)
            
    #solve diffeq for P_odor = 0
    #first, reinitialize lastnoise & noise
    noise = np.zeros((Ndim,1))
    lastnoise = np.zeros((Ndim,1))
    lastnoise = lastnoise + t_inh - noisewidth
    
    sol0 = sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
                    noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                    Sy2,Sx,Sx2,th,H0,W0,P_odor0,Ic,Ib),\
                    [0,395],init0,t_eval = teval,method = 'RK45')
    y0 = sol0.y
    y0 = np.transpose(y0)
    y0out = np.copy(y0)
    
    #convert signal into output signal given by the activation fn
    if signalflag ==1:
        for i in np.arange(np.size(t)):
            y0out[i,:Nmitral] = cellout(y0[i,:Nmitral],Sx,Sx2,th)
            y0out[i,Nmitral:] = cellout(y0[i,Nmitral:],Sy,Sy2,th)
    
    
    
    #*****************************************************************************
    
    #SIGNAL PROCESSING
    
    #Filtering the signal - O_mean: Lowpass fitered signal, under 20 Hz
    #S_h: Highpass filtered signal, over 20 Hz
    
    fs = 1/(.001*(t[1]-t[0]))  #sampling freq, converting from ms to sec
    
    f_c = 15/fs     # Cutoff freq at 20 Hz, written as a ratio of fc to sample freq
    
    flter = np.sinc(2*f_c*(t - (finalt-1)/2))*np.blackman(finalt) #creating the
                                                        #windowed sinc filter
                                                        #centered at the middle
                                                        #of the time data
    flter = flter/np.sum(flter)  #normalize
    
    hpflter = -np.copy(flter)
    hpflter[int((finalt-1)/2)] += 1  #convert the LP filter into a HP filter
    
    Sh = np.zeros(np.shape(yout))
    Sl = np.copy(Sh)
    Sl0 = np.copy(Sh)
    Sbp = np.copy(Sh)
    
    for i in np.arange(Ndim):
        Sh[:,i] = np.convolve(yout[:,i],hpflter,mode='same')
        Sl[:,i] = np.convolve(yout[:,i],flter,mode='same')
        Sl0[:,i] = np.convolve(y0out[:,i],flter,mode='same')
        
    #find the oscillation period Tosc (Tosc must be greater than 5 ms to exclude noise)
    Tosc0 = np.zeros(np.size(np.arange(5,50)))
    for i in np.arange(5,50):
        Sh_shifted=np.roll(Sh,i,axis=0)
        Tosc0[i-5] = np.sum(np.diagonal(np.dot(np.transpose(Sh[:,:Nmitral]),Sh_shifted[:,:Nmitral])))
        #That is, do the correlation matrix (time correlation), take the diagonal to
        #get the autocorrelations, and find the max
    Tosc = np.argmax(Tosc0)
    Tosc = Tosc + 5
    
    f_c2 = 1000*(1.3/Tosc)/fs  #Filter out components with frequencies higher than this
                            #to get rid of noise effects in cross-correlation
                            #times 1000 to get units right
    
    flter2 = np.sinc(2*f_c2*(t - (finalt-1)/2))*np.blackman(finalt)
    flter2 = flter2/np.sum(flter2)
    
    for i in np.arange(Ndim):
        Sbp[:,i] = np.convolve(Sh[:,i],flter2,mode='same')
    
    
    
    #CALCULATE THE DISTANCE MEASURES
    
    #calculate phase via cross-correlation with each cell
    phase = np.zeros(Nmitral)
    
    for i in np.arange(1,Nmitral):
        crosscor = signal.correlate(Sbp[:,0],Sbp[:,i])
        tdiff = np.argmax(crosscor)-(finalt-1)
        phase[i] = tdiff/Tosc * 2*np.pi
        
    #Problem with the method below is that it will only give values from 0 to pi
    #for i in np.arange(1,Nmitral):
    #    phase[i]=np.arccos(np.dot(Sbp[:,0],Sbp[:,i])/(lin.norm(Sbp[:,0])*lin.norm(Sbp[:,i])))
    
    OsciAmp = np.zeros(Nmitral)
    Oosci = np.copy(OsciAmp)*0j
    Omean = np.zeros(Nmitral)
    
    for i in np.arange(Nmitral):
        OsciAmp[i] = np.sqrt(np.sum(Sh[125:250,i]**2)/np.size(Sh[125:250,i]))
        Oosci[i] = OsciAmp[i]*np.exp(1j*phase[i])
        Omean[i] = np.average(Sl[:,i] - Sl0[:,i])
    
    Omean = np.maximum(Omean,0)
    
    Ooscibar = np.sqrt(np.dot(Oosci,np.conjugate(Oosci)))/Nmitral #can't just square b/c it's complex
    Omeanbar = np.sqrt(np.dot(Omean,Omean))/Nmitral
    
    
#    #######Bandpass-filter the output for beta and for gamma
#    
#    f_b_low = 15/fs
#    f_b_high = 34/fs
#    
#    f_gam_low = 40/fs
#    f_gam_high = 79/fs
#    
#    
#    #lowpass below the upper bound
#    fltr_b_high = np.sinc(2*f_b_high*(t - (finalt-1)/2))*np.blackman(finalt) 
#    fltr_b_high = fltr_b_high/np.sum(fltr_b_high)  #normalize
#    #highpass above the lower bound
#    fltr_b_low = np.sinc(2*f_b_low*(t - (finalt-1)/2))*np.blackman(finalt) 
#    fltr_b_low = -fltr_b_low/np.sum(fltr_b_low)  #normalize and make negative
#    fltr_b_low[int((finalt-1)/2)] += 1  #convert the LP filter into a HP filter
#    
#    fltr_gam_high = np.sinc(2*f_gam_high*(t - (finalt-1)/2))*np.blackman(finalt) 
#    fltr_gam_high = fltr_gam_high/np.sum(fltr_gam_high)  #normalize
#    
#    fltr_gam_low = np.sinc(2*f_gam_low*(t - (finalt-1)/2))*np.blackman(finalt) 
#    fltr_gam_low = -fltr_gam_low/np.sum(fltr_gam_low)  #normalize and make negative
#    fltr_gam_low[int((finalt-1)/2)] += 1  #convert the LP filter into a HP filter
#        
#    Sbp_b = np.zeros(np.shape(yout))
#    Sbp_gam = np.copy(Sbp_b)
#    
#    for i in np.arange(Ndim):
#        Sbp_b[:,i] = np.convolve(yout[:,i],fltr_b_low,mode='same')
#        Sbp_gam[:,i] = np.convolve(yout[:,i],fltr_gam_low,mode='same')
#    
#    for i in np.arange(Ndim):
#        Sbp_b[:,i] = np.convolve(Sbp_b[:,i],fltr_b_high,mode='same')
#        Sbp_gam[:,i] = np.convolve(Sbp_gam[:,i],fltr_gam_high,mode='same')
#        
#    
#    
#    
#    

    maxlam = np.max(np.abs(np.imag(np.sqrt(dA))))    
    
        
    return yout,y0out,Sh,t,OsciAmp,Omean,Oosci,Omeanbar,Ooscibar,dominant_freq,maxlam








def dmg_seed_50_2D(colnum):

#INITIALIZING STUFF
    Nmitral = 50
    Ngranule = np.copy(Nmitral) #number of granule cells     pg. 383 of Li/Hop
    Ndim = Nmitral+Ngranule #total number of cells
#    t_inh = 25 ; # time when inhalation starts
#    t_exh = 205; #time when exhalation starts
    
#    Ndamagetotal = Nmitral*2 + 1  #number of damage steps
    Ndamage = 18
#    Ncols = int(Nmitral/10)     #define number of columns to damage
    
    finalt = 395; # end time of the cycle
    
    #y = zeros(ndim,1);
                              
   
    
    P_odor0=np.zeros((Nmitral,1)) #odor pattern, no odor
    P_odor1 = P_odor0 + .00429   #Odor pattern 1
#    P_odor2 = 1/70*np.array([.6,.5,.5,.5,.3,.6,.4,.5,.5,.5])
#    P_odor3 = 4/700*np.array([.7,.8,.5,1.2,.7,1.2,.8,.7,.8,.8])
    #control_odor = control_order + .00429
    
    #control_odor = np.zeros((Nmitral,1)) #odor input for adaptation
    
    #controllevel = 1 #1 is full adaptation
    
    H0 = np.zeros((Nmitral,Ngranule)) #weight matrix: to mitral from granule
    W0 = np.zeros((Ngranule,Nmitral)) #weights: to granule from mitral
    
    
    H0 = np.load('H0_50_2D_60Hz.npy')   #load weight matrix
    
    
    
    W0 = np.load('W0_50_2D_60Hz.npy')  #load weight matrix
    
    W0_tot = np.sum(W0)         #get sum of the weights
    
    #H0 = H0 + H0*np.random.rand(np.shape(H0))
    #W0 = W0+W0*np.random.rand(np.shape(W0)) 
    
    M = 5   #average over 5 trials for each level of damage
    
    
    #initialize iterative variables
    d1it,d2it,d3it,d4it = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
    IPRit,IPR2it,pnit = np.zeros(M),np.zeros(M),np.zeros(M)
    frequencyit = np.zeros(M)
    pwrit,pwr_bit,pwr_gamit = np.zeros(M),np.zeros(M),np.zeros(M)
    yout2,Sh2 = np.zeros((finalt,Ndim)),np.zeros((finalt,Ndim))
#    Sbp_b,Sbp_gam = np.zeros((finalt,Ndim)),np.zeros((finalt,Ndim))
    psi = np.copy(Sh2[:,:Nmitral])
    
    #initialize quantities to be returned at end of the process
    dmgpct1 = np.zeros(Ndamage)
    eigfreq1 = np.zeros(Ndamage)
    d11 = np.zeros(Ndamage)
    d21 = np.zeros(Ndamage)
    d31 = np.zeros(Ndamage)
    d41 = np.zeros(Ndamage)
    pwr1 = np.zeros(Ndamage)
#    pwr_b,pwr_gam = np.zeros(Ndamage),np.zeros(Ndamage)
    IPR1 = np.zeros(Ndamage)
    IPR2 = np.zeros(Ndamage)
    pn1 = np.zeros(Ndamage)
    freq1 = np.zeros(Ndamage)
    cell_act = np.zeros((finalt,Ndim,Ndamage))
        
    Wdamagestep = .1*W0
    Wdamaged = np.copy(W0)
#    spread = -1 #start at -1 so that the first damage level has a spread of 0 radius
    damage = 0
    dmgcols = [colnum]
    
    #Get the base response first
    Omean1,Oosci1,Omeanbar1,Ooscibar1 = np.zeros((Nmitral,M))+0j,\
                np.zeros((Nmitral,M))+0j,np.zeros(M)+0j,np.zeros(M)+0j
    for m in np.arange(M):
        yout,y0out,Sh,t,OsciAmp1,Omean1[:,m],Oosci1[:,m],Omeanbar1[m],\
            Ooscibar1[m],freq0,maxlam = olf_bulb_10(Nmitral,H0,W0,P_odor1)
    
    for lv in np.arange(Ndamage):
        #reinitialize all iterative variables to zero (really only need to do for distance measures, but good habit)
        d1it,d2it,d3it,d4it = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
        IPRit,IPR2it,pnit = np.zeros(M),np.zeros(M),np.zeros(M)
        frequencyit = np.zeros(M)
        pwrit = np.zeros(M)
        
        if lv>0: 
            if lv>1:
                newcols = []
                for i in dmgcols:
                    newcols.extend([int(np.mod(i+1,5)),int(np.mod(i-1,5)),\
                           int(np.mod(i+5,Nmitral)),int(np.mod(i-5,Nmitral))])
                for n in newcols:
                    if not(n in dmgcols):
                        dmgcols.extend([n])
            Wdamaged[:,dmgcols] = Wdamaged[:,dmgcols]-Wdamagestep[:,dmgcols]
            Wdamaged[Wdamaged<0]=0
            damage = np.sum(W0-Wdamaged)
    
        for m in np.arange(M):
            #Then get respons of damaged network
            yout2[:,:],y0out2,Sh2[:,:],t2,OsciAmp2,Omean2,Oosci2,Omeanbar2,\
            Ooscibar2,freq2,grow_eigs2 = olf_bulb_10(Nmitral,H0,Wdamaged,P_odor1)
            #calculate distance measures
            print(time.time()-tm1)
            for i in np.arange(M):
                d1it[m] += 1-Omean1[:,m].dot(Omean2)/(lin.norm(Omean1[:,m])*lin.norm(Omean2))
                d2it[m] += 1-lin.norm(Oosci1[:,m].dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1[:,m])*lin.norm(Oosci2))
                d3it[m] += (Omeanbar1[m] - Omeanbar2)/(Omeanbar1[m] + Omeanbar2)
                d4it[m] += np.real((Ooscibar1[m] - Ooscibar2)/(Ooscibar1[m] + Ooscibar2))
            
            d1it[m] = d1it[m]/M
            d2it[m] = d2it[m]/M
            d3it[m] = d3it[m]/M
            d4it[m] = d4it[m]/M
            
            #calculate spectral density and "wave function" to get average power and IPR
            P_den = np.zeros((501,Nmitral))  #only calculate the spectral density from
            for i in np.arange(Nmitral):    #t=125 to t=250, during the main oscillations
                f, P_den[:,i] = signal.periodogram(Sh2[125:250,i],nfft=1000,fs=1000)
            psi = np.zeros(Nmitral)
            for p in np.arange(Nmitral):
                psi[p] = np.sum(P_den[:,p])
            psi = psi/np.sqrt(np.sum(psi**2))
            
#            P_den_b = np.zeros((501,Nmitral))
#            for i in np.arange(Nmitral):
#                f, P_den_b[:,i] = signal.periodogram(Sbp_b[125:250,i],nfft=1000,fs=1000)  #periodogram returns a list of
#                                                             #frequencies and the power density
#            P_den_gam = np.zeros((501,Nmitral))
#            for i in np.arange(Nmitral):
#                f, P_den_gam[:,i] = signal.periodogram(Sbp_gam[125:250,i],nfft=1000,fs=1000)  #periodogram returns a list of
#                                                 #frequencies and the power density
            
            psi2 = np.copy(OsciAmp2)
            psi2 = psi2/np.sqrt(np.sum(psi2**2))
            
            maxAmp = np.max(OsciAmp2)
            pnit[m] = len(OsciAmp2[OsciAmp2>maxAmp/2])
            
            IPRit[m] = 1/np.sum(psi**4)   
            IPR2it[m] = 1/np.sum(psi2**4)
            pwrit[m] = np.sum(P_den)/Nmitral
#            pwr_bit[m] = np.sum(P_den_b)/Nmitral
#            pwr_gamit[m] = np.sum(P_den_gam)/Nmitral
            
            #get the frequency according to the adiabatic analysis
            maxargs = np.argmax(P_den,axis=0) 
            argf = stats.mode(maxargs[maxargs!=0])
            frequencyit[m] = f[argf[0][0]]
#            print(cols)
#            print(time.time()-tm1)
#        
#        print('level',lv)
        #Get the returned variables for each level of damage
        dmgpct1[lv]=damage/W0_tot
        IPR1[lv] = np.average(IPRit)      
        pwr1[lv] = np.average(pwrit)
#        pwr_b[lv] = np.average(pwr_bit)
#        pwr_gam[lv] = np.average(pwr_gamit)
        freq1[lv]=np.average(frequencyit)         #0,1,2,3,4...Ndamage-1, then
                                                            #col 1 damage level 0,1,2...
#        IPRsd[lv]=np.std(IPRit)
#        pwrsd[lv]=np.std(pwrit)
#        freqsd[lv]=np.std(frequencyit)
        IPR2[lv] = np.average(IPR2it)
        pn1[lv] = np.average(pnit)
        
        d11[lv]= np.average(d1it)
        d21[lv] = np.average(d2it)
        d31[lv] = np.average(d3it)
        d41[lv] = np.average(d4it)
#        d1sd[lv] =  np.std(d1it)
#        d2sd[lv] = np.std(d2it)
#        d3sd[lv]=np.std(d3it)
#        d4sd[lv]=np.std(d4it)
        
        eigfreq1[lv] = np.copy(freq2)
        if arrayid == 0 or int(Nmitral/2):
            cell_act[:,:,lv]=np.copy(yout2)
   
    return dmgpct1,eigfreq1,d11,d21,d31,d41,pwr1,IPR1,IPR2,pn1,freq1,cell_act
    
    
    #save all the data
  
    
    
    
    
    

#******************************************************************************

Ndamage0 = 21  #Recording 0 level damage, too, so it will be 21 levels of damage

lvl = 0 #used to be an iterative variable, but now just a place holder
    
if __name__ == '__main__':
#    dmgpct,eigfreq = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    d1, d1sd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    d2,d2sd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    d3,d3sd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    d4,d4sd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    pwr,pwrsd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    IPR,IPRsd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
#    freq,freqsd = np.zeros((Ndamage,Nmitral)),np.zeros((Ndamage,Nmitral))
    
#    poolsize = np.copy(Nmitral)
#    Ncolumns = np.copy(Nmitral)
    Ncolumns = np.copy(Nmitral0)
    
    arrayid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    
    dmgpct,eigfreq,d1,d2,d3,d4,pwr,IPR,IPR2,pn,freq,cell_act = dmg_seed_50_2D(arrayid)  
    

    print(time.time()-tm1)
##************For testing the function*********************
    
#d1,d2,d3,d4 = [0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)]
#pwr,IPR,freq= [0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)]
#d1sd,d2sd,d3sd=[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)]
#d4sd,pwrsd,IPRsd = [0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)],[0 for i in range(Ndamage*Ncols)]
#freqsd,lock = [0 for i in range(Ndamage*Ncols)],0
#lvl,coln,dmgpct = 1,1,[0 for i in range(Ndamage*Ncols)]
#dmg_col_10_1D(lvl,coln,lock,dmgpct,d1,d2,d3,d4,pwr,IPR,freq,d1sd,d2sd,d3sd,d4sd,\
#                                                         pwrsd,IPRsd,freqsd)      



#******************************************************************************
    dmgpctfl,d1fl,d2fl,d3fl,d4fl,IPRfl,pwrfl,frequencyfl,eigfreqfl = "dmgpct","d1",\
    "d2","d3","d4","IPR","pwr","frequency","eigfreq"
#    d1sdfl,d2sdfl,d3sdfl,d4sdfl,IPRsdfl,pwrsdfl,frequencysdfl = "d1sd",\
#    "d2sd","d3sd","d4sd","IPRsd","pwrsd","frequencysd"
    IPR2fl,pnfl = "IPR2","pn"
    pd_type = "_50_" + str(arrayid)
    np.save(d1fl+pd_type,d1),np.save(d2fl+pd_type,d2),np.save(d3fl+pd_type,d3),np.save(d4fl+pd_type,d4)
    np.save(IPRfl+pd_type,IPR),np.save(pwrfl+pd_type,pwr)
#    np.save(d1sdfl+pd_type,d1sd),np.save(d2sdfl+pd_type,d2sd),np.save(d3sdfl+pd_type,d3sd),np.save(d4sdfl+pd_type,d4sd)
#    np.save(IPRsdfl+pd_type,IPRsd),np.save(pwrsdfl+pd_type,pwrsd)
    np.save(eigfreqfl+pd_type,eigfreq),np.save(dmgpctfl+pd_type,dmgpct)
    np.save(frequencyfl+pd_type,freq)
    np.save(IPR2fl+pd_type,IPR2), np.save(pnfl+pd_type,pn)
#    np.save(frequencysdfl+pd_type,freqsd)
#    np.save('pwr_b'+pd_type,pwr_b)
#    np.save('pwr_gam'+pd_type,pwr_gam)
    if arrayid == 0 or int(Nmitral0/2):
        np.save("cell_act"+pd_type,cell_act)