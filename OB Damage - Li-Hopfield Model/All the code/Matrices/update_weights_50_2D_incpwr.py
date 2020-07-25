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

#THIS SCRIPT IS FOR FINDING A WEIGHT MATRIX VIA RANDOM UPDATES                                

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

#plt.close('all')


#INITIALIZING STUFF

Nmitral = 50 #number of mitral cells
Ngranule = 50 #number of granule cells     pg. 383 of Li/Hop
Ndim = Nmitral+Ngranule #total number of cells
t_inh = 25 ; # time when inhalation starts
t_exh = 205; #time when exhalation starts
finalt = 395; # end time of the cycle
#    inity = np.zeros((Ndim,1)); # initial values of the neural internal state :
                          # inity[1:nmitral] is the initial values for the 
                          #mitral cells and inity[nmitral+1:end] is the initial
                          #values for the granule cells. Close to the equilibrium value
#    resty = np.zeros((Ndim,1)); # equilibrium value of the neural internal state when
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
#    P_odor2 = 1/70*np.array([.6,.5,.5,.5,.3,.6,.4,.5,.5,.5])
#    P_odor3 = 4/700*np.array([.7,.8,.5,1.2,.7,1.2,.8,.7,.8,.8])
#    
#    P_odor4 = 1/70*np.array([1.2,.1,1,.3,.8,.3,.5,.7,1.1,.8])
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
                              
#Icdecay = .03 #decay constant for control at exhalation
#yescontrol=0 #yescontrol = 1 for central control

signalflag = 1 # 0 for linear output, 1 for activation function



flnameH = "H0_50_2D"
flnameW = "W0_50_2D"

noise = np.zeros((Ndim,1)) #noise in inputs
noiselevel =  .00143
noisewidth = 7 #noise correlation time, given pg 383 Li/Hop as 9, but 7 in thesis

lastnoise = np.zeros((Ndim,1)) #initial time of last noise pule

#        Icontrol = np.zeros((Ngranule,1)) #central control signal for adaptation
                                #calculated when you call the derivative function

#Init_in = np.append(Ib,Ic,axis=1)
                                
                                
                                

#*****************************Set up 1D version of the matrix************

#H0_i = np.array([[.3,0,0,0,0,0,0,0,0,0],\
#               [.0,.4,0,0,0,0,0,0,0,0],\
#               [0,0,.3,0,0,0,0,0,0,0],\
#               [0,0,0,.5,0,0,0,0,0,0],\
#               [0,0,0,0,.3,0,0,0,0,0],\
#               [0,0,0,0,0,.3,0,0,0,0],\
#               [0,0,0,0,0,0,.4,0,0,0],\
#               [0,0,0,0,0,0,0,.5,0,0],\
#               [0,0,0,0,0,0,0,0,.3,0],\
#               [0,0,0,0,0,0,0,0,0,.3]])
#
#H0 = np.zeros((Nmitral,Ngranule))
#diag_row = np.r_[0:Nmitral]
#diag_col = np.r_[0:Ngranule]
#left_diag_row = np.r_[1:Nmitral]
#left_diag_col = np.r_[0:Ngranule-1]
#right_diag_row = np.copy(left_diag_col)
#right_diag_col = np.copy(left_diag_row)
#
#entries = []
#for i in np.arange(Nmitral/10):
#    entries = np.append(entries,H0_i[H0_i>0])
#    
#H0[diag_row,diag_col] = np.copy(entries)
#
#
#W0_i = np.array([[.3,0,0,0,0,0,0,0,0,0],\
#                   [0,.2,0,0,0,0,0,0,0,0],\
#                   [0,0,.3,0,0,0,0,0,0,0],\
#                   [0,0,0,.2,0,0,0,0,0,0],\
#                   [0,0,0,0,.1,0,0,0,0,0],\
#                   [0,0,0,0,0,.3,0,0,0,0],\
#                   [0,0,0,0,0,0,.3,0,0,0],\
#                   [0,0,0,0,0,0,0,.3,0,0],\
#                   [0,0,0,0,0,0,0,0,.3,0],\
#                   [0,0,0,0,0,0,0,0,0,.5]])
#    
#W0 = np.zeros((Nmitral,Ngranule))

   
#
#entries = []
#for i in np.arange(Nmitral/10):
#    entries = np.append(entries,W0_i[W0_i>0])
#    
#W0[diag_row,diag_col] = np.copy(entries)

#****************** Make it 2D******************
#10 rows of 5 cols
 
#connections: i to i+5 mod 50, i to i-5 mod 50
    #first in row: (if i mod 4 = 0) i to i+3
    #last in row: (if i mod 4 = 3) i to i-3
    
#H0weights = np.r_[.3:1.1:.1]
#W0weights = np.r_[.1:.8:.1]
#
#nrows = 10
#ncols = 5
#
#H0[0,-1],H0[-1,0],W0[0,-1],W0[-1,0] = 0,0,0,0
#
#for i in np.arange(Nmitral):
#    for n in np.arange(Nmitral):
#        H0[i,np.mod(i+ncols,Nmitral)] = np.random.choice(H0weights) #connections "above" and
#        H0[i,np.mod(i-ncols,Nmitral)] = np.random.choice(H0weights) # "below"
#        W0[i,np.mod(i+ncols,Nmitral)] = np.random.choice(W0weights)
#        W0[i,np.mod(i-ncols,Nmitral)] = np.random.choice(W0weights)
#        #connections left and right
#        if np.mod(i,ncols)==0: #periodic: adjacent to right and loop left
#            H0[i,i+ncols-1],H0[i,i+1]=np.random.choice(H0weights),np.random.choice(H0weights)
#            W0[i,i+ncols-1],W0[i,i+1]=np.random.choice(W0weights),np.random.choice(W0weights)
#        elif np.mod(i,ncols)==ncols-1: #periodic: adjacent to left and loop right
#            H0[i,i-1],H0[i,i-(ncols-1)]=np.random.choice(H0weights),np.random.choice(H0weights)
#            W0[i,i-1],W0[i,i-(ncols-1)]=np.random.choice(W0weights),np.random.choice(W0weights)
#        else:                   #middle: just adjacent left and right
#            H0[i,i+1],H0[i,i-1] = np.random.choice(H0weights),np.random.choice(H0weights)
#            W0[i,i+1],W0[i,i-1] = np.random.choice(W0weights),np.random.choice(W0weights)

H0 = np.load("H0_50_2D_ipr.npy")
W0 = np.load("W0_50_2D_ipr.npy")

H0index = np.where(H0>0)
W0index = np.where(W0>0)

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

#        x0 = np.zeros((Ndim,1))
xequi = fsolve(lambda x: equi(x,Ndim,Nmitral,Sx,Sx2,Sy,Sy2,th,alpha,\
                                 t_inh,H0,W0,P_odor1,Ib,Ic),rest0) 
                    #equilibrium values with some input, about 20 ms to run





#*****************************************************************************

#SIGNAL PROCESSING

#Filtering the signal - O_mean: Lowpass fitered signal, under 20 Hz
#S_h: Highpass filtered signal, over 20 Hz

t = np.r_[0:finalt]  

fs = 1/(.001*(t[1]-t[0]))  #sampling freq, converting from ms to sec

f_c = 40/fs     # Cutoff freq at 20 Hz, written as a ratio of fc to sample freq

flter = np.sinc(2*f_c*(t - (finalt-1)/2))*np.blackman(finalt) #creating the
                                                    #windowed sinc filter
                                                    #centered at the middle
                                                    #of the time data
flter = flter/np.sum(flter)  #normalize

hpflter = -np.copy(flter)
hpflter[int((finalt-1)/2)] += 1  #convert the LP filter into a HP filter



#*************************************************************************

#SOLVE DIFFERENTIAL EQUATIONS TO GET INPUT AND OUTPUTS AS FN'S OF t
  
    #differential equation to solve
teval = np.r_[0:finalt]
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
        
#first, reinitialize lastnoise & noise
noise = np.zeros((Ndim,1))
lastnoise = np.zeros((Ndim,1))
lastnoise = lastnoise + t_inh - noisewidth


Sh = np.zeros(np.shape(yout))


for i in np.arange(Ndim):
    Sh[:,i] = np.convolve(yout[:,i],hpflter,mode='same')

#Calculate IPR
P_den = np.zeros((198,Ndim))
for i in np.arange(Nmitral):
    f, P_den[:,i] = signal.periodogram(Sh[:,i])  #periodogram returns a list of
                                                 #frequencies and the power density
psi = np.zeros(Nmitral)
for i in np.arange(Nmitral):
    psi[i] = np.sum(P_den[:,i])
psi = psi/np.sqrt(np.sum(psi**2))
IPR1 = 1/np.sum(psi**4)

pwr = np.sum(P_den)/Nmitral

print(IPR1)

updated = 0  

###############Start Updating#########################################
while pwr < .15:
    
    H0trial = np.copy(H0)
    W0trial = np.copy(W0)
    
    #Add in random number between -.05 and .05 to each of the nonzero entries of
    #H0 and W0
    for i in np.arange(np.size(H0trial[H0trial>0])):
        H0trial[H0index[0][i],H0index[1][i]] += .2*(1-2*np.random.random())
        W0trial[W0index[0][i],W0index[1][i]] += .2*(1-2*np.random.random())
        
    H0trial[H0trial<0] = .05  #Just in case anything goes negative
    W0trial[W0trial<0] = .05
    
    #Now re-solve the equations with the trial weights
    sol = solve_ivp(lambda t,y: diffeq(t,y,Nmitral,Ngranule,Ndim,lastnoise,\
                    noise,noisewidth,noiselevel, t_inh,t_exh,exh_rate,alpha,Sy,\
                    Sy2,Sx,Sx2,th,H0trial,W0trial,P_odor1,Ic,Ib),\
                    [0,395],init0,t_eval = teval,method = 'RK45') 
    t = sol.t
    ytrial = sol.y
    ytrial = np.transpose(ytrial)
    youttrial = np.copy(ytrial)
    
    #convert signal into output signal given by the activation fn
    if signalflag ==1:
        for i in np.arange(np.size(t)):
            youttrial[i,:Nmitral] = cellout(ytrial[i,:Nmitral],Sx,Sx2,th)
            youttrial[i,Nmitral:] = cellout(ytrial[i,Nmitral:],Sy,Sy2,th)
            

    #first, reinitialize lastnoise & noise
    noise = np.zeros((Ndim,1))
    lastnoise = np.zeros((Ndim,1))
    lastnoise = lastnoise + t_inh - noisewidth
    
    
    Shtrial = np.zeros(np.shape(youttrial))
    
    
    for i in np.arange(Ndim):
        Shtrial[:,i] = np.convolve(youttrial[:,i],hpflter,mode='same')
    
    #Calculate the power spectrum using the trial weights
    P_dentrial = np.zeros((198,Ndim))
    for i in np.arange(Nmitral):
        f, P_dentrial[:,i] = signal.periodogram(Shtrial[:,i])  #periodogram returns a list of
                                                     #frequencies and the power density
    osc_power2 = np.sum(P_dentrial)/Nmitral
    
    psi = np.zeros(Nmitral)
    for i in np.arange(Nmitral):
        psi[i] = np.sum(P_dentrial[:,i])
    psi = psi/np.sqrt(np.sum(psi**2))
    IPR2 = 1/np.sum(psi**4)
    print(IPR2)
    print(osc_power2)
    #if the new weights lead to better oscillations, then keep the changes
    if osc_power2>pwr and IPR2>10:
        H0 = np.copy(H0trial)
        W0 = np.copy(W0trial)
        updated = updated+1
        print('updated')
        IPR1 = np.copy(IPR2)
        pwr = np.copy(osc_power2)


#        timeelapsed = time.time()-time1
#        print(updated)
#        print(timeelapsed)
#        print(p)

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
#        candidates = [np.real((dA[indices])**.5),np.real((dA[indices2])**.5)]
largest = np.argmax(diff_re)

check = np.size(indices)
check2 = np.size(indices2)


if check==0 and check2==0:
    print("No Odor Recognized")

else:
#    dominant_freq = np.max(candidates)/(2*np.pi) #find frequency of the dominant mode
                                            #Divide by 2pi to get to cycles/ms
    dominant_freq = np.real((dA[largest])**.5)/(2*np.pi)
#            print("Odor detected. Eigenvalues:",dA[indices],dA[indices2],\
#                  "\nEigenvectors:",vA[indices],vA[indices2],\
#                  "\nDominant Frequency:",dominant_freq)

        
np.save(flnameH,H0)
np.save(flnameW,W0)


#******************************************************************************

#PLOT STUFF
 
#fig1 = plt.figure(1,figsize = (16,10)) #figsize (26,12) is full screen
#fig1.canvas.set_window_title('cell activity')  #cell outputs
#fig1.suptitle('Cell Activity, output vs. time')
#N_plot_col = Nmitral/2
#for i in np.arange(Nmitral):
#    plt.subplot(N_plot_col,2,i+1)
#    plt.plot(t,youttrial[:,i])
#    plt.ylim(0,np.max(youttrial[:,:Nmitral]))
##for i in np.arange(Ngranule):
##    plt.subplot(5,2,i+1)
##    plt.plot(t,yout[:,i+Nmitral])
#
##fig2 = plt.figure(2,figsize = (16,10))  #Plot highpass filtered outputs for each cell
##fig2.suptitle('Cell Activity - Highpass Filtered, output vs. time')
##for i in np.arange(Nmitral):
##    plt.subplot(N_plot_col,2,i+1)
##    plt.plot(t,Sh[:,i])
##    plt.ylim(np.min(Sh[:,:Nmitral]),np.max(Sh[:,:Nmitral]))
##for i in np.arange(Ngranule):
##    plt.subplot(5,2,i+1)
##    plt.plot(t,Sh[:,i+Nmitral])
#    
##fig3 = plt.figure(3,figsize = (16,10))  #Plot the fourier transform of each output
##fig3.suptitle('Fourier Transform, signal vs. frequency')
##Shfreq = np.fft.rfftfreq(np.size(Sh[:,0]))  #get freq points for FT plots
##for i in np.arange(Nmitral):
##    plt.subplot(5,2,i+1)
##    plt.plot(Shfreq,np.fft.rfft(Sh[:,i]))  #plot the Fourier Transform of each
##    plt.xlim(0,.2)
##for i in np.arange(Ngranule):               #cell's activity
##    plt.subplot(5,2,i+1)
##    plt.plot(Shfreq,np.fft.rfft(Sh[:,i+Nmitral]))
##    plt.xlim(0,.2)
##plt.xlabel('Frequency (kHz) Theoretical Dominant Freq = ' + str(dominant_freq))
#
#fig4 = plt.figure(4, figsize = (16,10))  #Plot the periodogram for each output
#fig4.suptitle('Periodogram, power density vs. frequency')
#for i in np.arange(Nmitral):                   
#    plt.subplot(N_plot_col,2,i+1)                    
#    plt.plot(f,P_dentrial[:,i])
#    plt.xlim(0,.2)
#    plt.ylim(0,np.max(P_dentrial[:,:Nmitral]))
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

