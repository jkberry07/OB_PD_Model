# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:38:32 2018

@author: wmmjk
"""

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
import time

plt.close('all')


tm0 = time.time()
#INITIALIZING STUFF

Nmitral = 100 #number of mitral cells
Ngranule = 100 #number of granule cells     pg. 383 of Li/Hop
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




H0_10 = np.array([[.3,.9,0,0,0,0,0,0,0,.7],\
               [.9,.4,1,0,0,0,0,0,0,0],\
               [0,.8,.3,.8,0,0,0,0,0,0],\
               [0,0,.7,.5,.9,0,0,0,0,0],\
               [0,0,0,.8,.3,.8,0,0,0,0],\
               [0,0,0,0,.7,.3,.9,0,0,0],\
               [0,0,0,0,0,.7,.4,.9,0,0],\
               [0,0,0,0,0,0,.5,.5,.7,0],\
               [0,0,0,0,0,0,0,.9,.3,.9],\
               [.9,0,0,0,0,0,0,0,.8,.3]])

H0 = np.zeros((100,100))

for i in np.r_[0:91:10]:
    H0[i:i+10,i:i+10] = H0_10

for i in np.r_[10:81:10]:
    H0[i,i+9],H0[i+9,i] = 0,0

H0[0,99] = H0_10[0,9]
H0[99,0] = H0_10[9,0]
H0[9,10],H0[90,89] = H0_10[9,0],H0_10[0,9]
H0[0,9],H0[9,0],H0[90,99],H0[99,90] = 0,0,0,0

for i in np.r_[10:91:10]:
    H0[i,i-1],H0[i-1,i] = H0_10[0,9],H0_10[9,0]

H0[H0>0] = H0[H0>0] + .1*(2*np.random.rand(np.count_nonzero([H0>0]))-1)

W0_10 = np.array([[.3,.7,0,0,0,0,0,0,.5,.3],\
               [.3,.2,.5,0,0,0,0,0,0,.7],\
               [0,.1,.3,.5,0,0,0,0,0,0],\
               [0,.5,.2,.2,.5,0,0,0,0,0],\
               [.5,0,0,.5,.1,.9,0,0,0,0],\
               [0,0,0,0,.3,.3,.5,.4,0,0],\
               [0,0,0,.6,0,.2,.3,.5,0,0],\
               [0,0,0,0,0,0,.5,.3,.5,0],\
               [0,0,0,0,0,.2,0,.2,.3,.7],\
               [.7,0,0,0,0,0,0,.2,.3,.5]])
    
W0 = np.zeros((100,100))

for i in np.r_[0:91:10]:
    W0[i:i+10,i:i+10] = W0_10

for i in np.r_[10:81:10]:
    W0[i,i+9],W0[i+9,i] = 0,0

W0[0,99] = W0_10[0,9]
W0[99,0] = W0_10[9,0]
W0[9,10],W0[90,89] = W0_10[9,0],W0_10[0,9]
W0[0,9],W0[9,0],W0[90,99],W0[99,90] = 0,0,0,0

for i in np.r_[10:91:10]:
    W0[i,i-1],W0[i-1,i] = W0_10[0,9],W0_10[9,0]

W0[W0>0] = W0[W0>0] + .1*(2*np.random.rand(np.count_nonzero([W0>0]))-1)

#H0 = H0 + H0*np.random.rand(np.shape(H0))
#W0 = W0+W0*np.random.rand(np.shape(W0))

#H0[:,4] = 0



#******************************************************************************

#CALCULATE FIXED POINTS

#Calculating equilibrium value with no input
rest0 = np.zeros((Ndim,1))

restequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                 t_inh,H0,W0,P_odor0,Ib,Ic),rest0) #about 20 ms to run this

np.random.seed(seed=23)
init0 = restequi+np.random.rand(Ndim)*.00143 #initial conditions plus some noise
                                           #for no odor input
np.random.seed()
#Now calculate equilibrium value with odor input
                                            

lastnoise = lastnoise + t_inh - noisewidth  #initialize lastnoise value
                                            #But what is it for? to have some
                                            #kind of correlation in the noise
                                            
#find eigenvalues of A to see if input produces oscillating signal

x0 = np.zeros((Ndim,1))
xequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                 t_inh,H0,W0,P_odor1,Ib,Ic),rest0) 
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
candidates = [np.real((dA[indices])**.5),np.real((dA[indices2])**.5)]

check = np.size(indices)
check2 = np.size(indices2)


if check==0 and check2==0:
    print("No Odor Recognized")
else:
    dominant_freq = np.max(candidates)/(2*np.pi) #find frequency of the dominant mode
                                            #Divide by 2pi to get to cycles/ms
    print("Odor detected. Eigenvalues:",dA[indices],dA[indices2],\
          "\nEigenvectors:",vA[indices],vA[indices2],\
          "\nDominant Frequency:",dominant_freq)






#*************************************************************************

#SOLVE DIFFERENTIAL EQUATIONS TO GET INPUT AND OUTPUTS AS FN'S OF t
  
    #differential equation to solve
teval = np.r_[0:finalt]                                              

#solve the differential equation
sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
                noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                Sy2,Sx,Sx2,th,H0,W0,P_odor1,Ic,Ib),\
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
t0 = sol0.t
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

f_c = 20/fs     # Cutoff freq at 20 Hz, written as a ratio of fc to sample freq

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

P_den = np.zeros((198,Ndim))
for i in np.arange(Nmitral):
    f, P_den[:,i] = signal.periodogram(Sh[:,i])  #periodogram returns a list of
                                                 #frequencies and the power density
osc_power1 = np.sum(P_den)/Nmitral



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
    OsciAmp[i] = np.sqrt(np.sum(Sh[:,i]**2)/np.size(Sh[:,i]))
    Oosci[i] = OsciAmp[i]*np.exp(1j*phase[i])
    Omean[i] = np.average(Sl[:,i] - Sl0[:,i])

Omean = np.maximum(Omean,0)

Ooscibar = np.sqrt(np.dot(Oosci,np.conjugate(Oosci)))/Nmitral #can't just square b/c it's complex
Omeanbar = np.sqrt(np.dot(Omean,Omean))/Nmitral

#d1 = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
#d2 = 1 - lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
#
#d3 = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
#d4 = (Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2)




tm1 = time.time()

tm = tm1-tm0
print(tm)

#******************************************************************************

#PLOT STUFF
 
#fig1 = plt.figure(1,figsize = (16,9)) #figsize (26,12) is full screen
#fig1.canvas.set_window_title('cell activity')  #cell outputs
#fig1.suptitle('Cell Activity, output vs. time')
for n in np.arange(10):
    plt.figure(n,figsize = (16,9))
    for i in np.arange(10):
        plt.subplot(5,2,i+1)
        plt.plot(t,yout[:,i + n*10])
        plt.ylim(0,np.max(yout[:,:Nmitral]))
#for i in np.arange(Ngranule):
#    plt.subplot(5,2,i+1)
#    plt.plot(t,yout[:,i+Nmitral])

#fig2 = plt.figure(2,figsize = (16,9))  #Plot highpass filtered outputs for each cell
#fig2.suptitle('Cell Activity - Highpass Filtered, output vs. time')
#for i in np.arange(Nmitral):
#    plt.subplot(5,2,i+1)
#    plt.plot(t,Sh[:,i])
#    plt.ylim(np.min(Sh[:,:Nmitral]),np.max(Sh[:,:Nmitral]))
#for i in np.arange(Ngranule):
#    plt.subplot(5,2,i+1)
#    plt.plot(t,Sh[:,i+Nmitral])
    
#fig3 = plt.figure(3,figsize = (16,10))  #Plot the fourier transform of each output
#fig3.suptitle('Fourier Transform, signal vs. frequency')
#Shfreq = np.fft.rfftfreq(np.size(Sh[:,0]))  #get freq points for FT plots
#for i in np.arange(Nmitral):
#    plt.subplot(5,2,i+1)
#    plt.plot(Shfreq,np.fft.rfft(Sh[:,i]))  #plot the Fourier Transform of each
#    plt.xlim(0,.2)
#for i in np.arange(Ngranule):               #cell's activity
#    plt.subplot(5,2,i+1)
#    plt.plot(Shfreq,np.fft.rfft(Sh[:,i+Nmitral]))
#    plt.xlim(0,.2)
#plt.xlabel('Frequency (kHz) Theoretical Dominant Freq = ' + str(dominant_freq))

#fig4 = plt.figure(4, figsize = (16,9))  #Plot the periodogram for each output
#fig4.suptitle('Periodogram, power density vs. frequency')
#P_den = np.zeros((198,Ndim))
##marker = np.zeros(np.shape(P_den[:,0]))
#for i in np.arange(Nmitral):
#    f, P_den[:,i] = signal.periodogram(Sh[:,i])  #periodogram returns a list of 
#    plt.subplot(5,2,i+1)                    #frequencies and the power density
#    plt.plot(f,P_den[:,i])
#    plt.xlim(0,.2)
#for i in np.arange(Ngranule):
#    f, P_den[:,i+Nmitral] = signal.periodogram(Sh[:,i+Nmitral])
#    plt.subplot(5,2,i+1)
#    plt.plot(f,P_den[:,i+Nmitral])
##    marker[17] = np.max(P_den[:,i+Nmitral])
##    plt.plot(f,marker)
#    plt.xlim(0,.2)
    

#plt.figure(5)  #Plot the spectrogram for each output
#for i in np.arange(Nmitral):
#    f2,t2,Spect = signal.spectrogram(Sh[:,i], nperseg = 30)   #Spectrogram returns a list of
#    plt.subplot(5,2,i+1)                        #frequencies, times, and a 2D
#    plt.pcolormesh(t2,f2,Spect)              #array of values for each f and t
#    plt.ylim(0,.2)
#plt.ylabel('Frequency (kHz)')
#plt.xlabel('time (ms)')
    

