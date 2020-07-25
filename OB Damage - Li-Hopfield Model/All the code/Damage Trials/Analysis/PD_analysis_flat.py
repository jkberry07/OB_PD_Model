# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:38:32 2018

@author: wmmjk
"""

#Olfactory Bulb Model a la Li/Hopfield and Li/Hertz
#Translated from Avinash's Matlab code
#This one I have the initial condition in solveivp as the equilibrium at rest plus noise

#Calculates phase differences, distance measure terms, high and low-passed output,
#power spectrum, IPR

#Change Log

#01/28/19   - added a variable "yout" so I could look at both internal state and 
#            output after solving ode
#
#01/29/19      

#08/2/19    - Was printing out the row of vA instead of the column. fixed now.        

import numpy as np
#import scipy.linalg as lin
#from scipy.integrate import solve_ivp
#from cellout import cellout,celldiff
#from scipy.optimize import fsolve
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from matplotlib.animation import FFMpegWriter
#import scipy.signal as signal
#from numpy.random import rand
#from olf_diffeq import diffeq
#from olf_equi import equi
#from olf_diffeq import testdiffeq


plt.close('all')

#INITIALIZING STUFF


Nmitral = 50 #number of mitral cells
Ngranule = 50 #number of granule cells     pg. 383 of Li/Hop

flpath = 'C:\\Users\\wmmjk\\.spyder-py3\\OI-flat-trials\\50_2D\\'
#flpath = 'C:\\Users\\wmmjk\\.spyder-py3\\'

num = "_"+str(Nmitral)
dmgtype = '_2D_flat'

Ndamage = 21#how many increments over which damage was administered
#Seed trials:
#10_1D: 16
#10_2D: 14
#20_1D: 21
#20_2D: 15
#50_1D: 36
#50_2D: 18
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



##********************* Load Variables *********************************

#H0 = np.load("H0_10_2D_60Hz.npy")
#W0 = np.load("W0_10_2D_60Hz.npy")
t = np.load("t.npy")


dmgpctfl,d1fl,d2fl,d3fl,d4fl,IPRfl,pwrfl,frequencyfl,eigfreqfl = "dmgpct","d1",\
    "d2","d3","d4","IPR","pwr","frequency","eigfreq"
d1sdfl,d2sdfl,d3sdfl,d4sdfl,IPRsdfl,pwrsdfl,frequencysdfl = "d1sd",\
    "d2sd","d3sd","d4sd","IPRsd","pwrsd","frequencysd"
IPR2fl,pnfl = "IPR2","pn"

#eigfreq,d1 = np.zeros((Nmitral,Ndamage)),np.zeros((Nmitral,Ndamage))
#d2,d3,d4 = np.zeros((Nmitral,Ndamage)),np.zeros((Nmitral,Ndamage)),np.zeros((Nmitral,Ndamage))
#IPR,pwr = np.zeros((Nmitral,Ndamage)),np.zeros((Nmitral,Ndamage))
#IPR2,pn = np.zeros((Nmitral,Ndamage)),np.zeros((Nmitral,Ndamage))
#frequency = np.zeros((Nmitral,Ndamage))
#damagelvlraw = np.zeros((Nmitral,Ndamage))


flend = '_'+str(1)+'.npy'
damagelvl = np.load(flpath+dmgpctfl+num+flend)
eigfreq = np.load(flpath+eigfreqfl+num+flend)

d1=np.load(flpath+d1fl+num+flend)
#d1sd=np.load(flpath+d1sdfl+flend)

d2=np.load(flpath+d2fl+num+flend)
#d2sd=np.load(flpath+d2sdfl+flend)

d3=np.load(flpath+d3fl+num+flend)
#d3sd=np.load(flpath+d3sdfl+flend)

d4=np.load(flpath+d4fl+num+flend)
#d4sd=np.load(flpath+d4sdfl+flend)

IPR=np.load(flpath+IPRfl+num+flend)
#IPRsd=np.load(flpath+IPRsdfl+flend)

pwr=np.load(flpath+pwrfl+num+flend)
#pwrsd=np.load(flpath+pwrsdfl+flend)

#pwr = pwr/pwr[0]

IPR2 = np.load(flpath+IPR2fl+num+flend)
pn = np.load(flpath+pnfl+num+flend)

frequency = np.load(flpath+frequencyfl+num+flend)
#frequencysd = np.load(flpath+frequencysdfl+flend)
#yout=np.load(yout2fl+flend)

#H0 = H0 + H0*np.random.rand(np.shape(H0))
#W0 = W0+W0*np.random.rand(np.shape(W0))

#H0[:,4] = 0

#******************************************************************************
#*******Process Variables*****************************************
d1avg,d1avgsd = np.zeros(Ndamage),np.zeros(Ndamage)
#damagelvl = np.zeros(Ndamage)
d2avg,d2avgsd = np.zeros(Ndamage),np.zeros(Ndamage)
d3avg,d3avgsd = np.zeros(Ndamage),np.zeros(Ndamage)
d4avg,d4avgsd = np.zeros(Ndamage),np.zeros(Ndamage)
IPRavg,IPRavgsd = np.zeros(Ndamage),np.zeros(Ndamage)
IPR2avg,IPR2avgsd = np.zeros(Ndamage),np.zeros(Ndamage)
pnavg,pnavgsd = np.zeros(Ndamage),np.zeros(Ndamage)
pwravg,pwravgsd = np.zeros(Ndamage),np.zeros(Ndamage)
frequencyavg,frequencyavgsd = np.zeros(Ndamage),np.zeros(Ndamage)
eigfreqavg,eigfreqavgsd = np.zeros(Ndamage), np.zeros(Ndamage)
#for i in np.arange(Ndamage):                                    #Averaging over all columns
#    d1avg[i] = np.average(d1[:,i])  #Give an array that picksout the ith level of damage
#    d1avgsd[i] = np.std(d1[:,i])
##    d1avgsd[i] = np.sqrt(np.sum(d1sd[:,i]**2))/Nmitral
#    d2avg[i] = np.average(d2[:,i]) #in each column
#    d2avgsd[i] = np.std(d2[:,i])
##    d2avgsd[i] = np.sqrt(np.sum(d2[:,i]**2))/Nmitral    
#    d3avg[i] = np.average(d3[:,i])
#    d3avgsd[i] = np.std(d3[:,i])
##    d3avgsd[i] = np.sqrt(np.sum(d3[:,i]**2))/Nmitral
#    d4avg[i] = np.average(d4[:,i])
#    d4avgsd[i] = np.std(d4[:,i])
##    d4avgsd[i] = np.sqrt(np.sum(d4[:,i]**2))/Nmitral
#    IPRavg[i] = np.average(IPR[:,i])
#    IPRavgsd[i] = np.std(IPR[:,i])
##    IPRavgsd[i] = np.sqrt(np.sum(IPR[:,i]**2))/Nmitral
#    pwravg[i] = np.average(pwr[:,i])
#    pwravgsd[i] = np.std(pwr[:,i])
##    pwravgsd[i] = np.sqrt(np.sum(pwr[:,i]**2))/Nmitral
#    IPR2avg[i] = np.average(IPR2[:,i])
#    IPR2avgsd[i] = np.std(IPR2[:,i])
#    pnavg[i] = np.average(pn[:,i])
#    pnavgsd[i] = np.std(pn[:,i])
#    
#    frequencyavg[i] = np.average(frequency[:,i])
#    frequencyavgsd[i] = np.std(frequency[:,i])
##    frequencyavgsd[i] = np.sqrt(np.sum(frequency[:,i]**2))/Nmitral
#    damagelvl[i] = np.average(damagelvlraw[:,i])
#    
#    eigfreqavg[i]=np.average(eigfreq[:,i])
#    eigfreqavgsd[i] = np.std(eigfreq[:,i])

#pwravgsd = pwravgsd/pwravg[0] #Have to do this one first, obviously
#pwravg = pwravg/pwravg[0]




#******************************************************************************
#PLOT STUFF
 
#fig1 = plt.figure(1,figsize = (16,10)) #figsize (26,12) is full screen
#fig1.canvas.set_window_title('cell activity')  #cell outputs
#fig1.suptitle('Cell Activity, output vs. time')
#for i in np.arange(Nmitral):
#    plt.subplot(Nmitral/2,2,i+1)
#    plt.plot(t,yout[:,i])
#    plt.ylim(0,np.max(yout[:,:Nmitral]))
#for i in np.arange(Ngranule):
#    plt.subplot(5,2,i+1)
#    plt.plot(t,yout[:,i+Nmitral])
    
np.save(flpath+'damagelvl.npy',damagelvl)

plt.rcParams.update({'font.size':22})

    
fig2 = plt.figure(2,figsize = (16,9))
fig2.canvas.set_window_title('d1')
fig2.suptitle('d1 vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,d1)
plt.plot(damagelvl,.15*np.ones(np.shape(damagelvl)),'g--')
plt.ylim(0,1.1)
plt.savefig(flpath+'d1'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'d1.npy',d1)
np.save(flpath+'d1sd.npy',d1avgsd)

fig3 = plt.figure(3,figsize = (16,9))
fig3.canvas.set_window_title('d2')
fig3.suptitle('d2 vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,d2)
plt.plot(damagelvl,.15*np.ones(np.shape(damagelvl)),'g--')
plt.ylim(0,1.1)
plt.savefig(flpath+'d2'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath + 'd2.npy',d2)
np.save(flpath+'d2sd.npy',d2avgsd)

fig4 = plt.figure(4,figsize = (16,9))
fig4.canvas.set_window_title('d3')
fig4.suptitle('d3 vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,d3)
plt.plot(damagelvl,.15*np.ones(np.shape(damagelvl)),'g--')
plt.ylim(-1.1,1.1)
plt.savefig(flpath+'d3'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'d3.npy',d3)
np.save(flpath+'d3sd.npy',d3avgsd)

fig5 = plt.figure(5,figsize = (16,9))
fig5.canvas.set_window_title('d4')
fig5.suptitle('d4 vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,d4)
plt.plot(damagelvl,.15*np.ones(np.shape(damagelvl)),'g--')
plt.ylim(-1.1,1.1)
plt.savefig(flpath+'d4'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'d4.npy',d4)
np.save(flpath+'d4sd.npy',d4avgsd)

fig6 = plt.figure(6,figsize = (16,9))
fig6.canvas.set_window_title('Oscillatory Power')
fig6.suptitle('Oscillatory Power vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,pwr)
plt.ylim(0,np.max(pwr)*1.5)
plt.savefig(flpath+'pwr'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'pwr.npy',pwr)
np.save(flpath+'pwrsd.npy',pwravgsd)

#fig7 = plt.figure(7,figsize = (16,9))
#fig7.canvas.set_window_title('IPR')
#fig7.suptitle('IPR vs. Damage level (increments of 5% undamaged weight)')
#plt.plot(damagelvl,IPR)
#plt.ylim(0,Nmitral)
#plt.savefig(flpath+'IPR'+num+dmgtype+'.png',bbox_inches='tight')

fig10 = plt.figure(10,figsize = (16,9))
fig10.canvas.set_window_title('IPR2 (Uses amplitude)')
fig10.suptitle('IPR2 vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,IPR2)
plt.ylim(0,Nmitral)
plt.savefig(flpath+'IPR2'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'IPR2.npy',IPR2)
np.save(flpath+'IPR2sd.npy',IPR2avgsd)

#fig11 = plt.figure(11,figsize = (16,9))
#fig11.canvas.set_window_title('PN (Number of mitral cells with amplitude above 50% of max)')
#fig11.suptitle('PN vs. Damage level (increments of 5% undamaged weight)')
#plt.plot(damagelvl,pn)
#plt.ylim(0,Nmitral)
#plt.savefig(flpath+'pn'+num+dmgtype+'.png',bbox_inches='tight')

fig8 = plt.figure(8,figsize = (16,9))
fig8.canvas.set_window_title('Frequency')
fig8.suptitle('Frequency vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,frequency)
plt.ylim(0,100)
plt.savefig(flpath+'freq'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'freq.npy',frequency)
np.save(flpath+'freqsd.npy',frequencyavgsd)

fig9 = plt.figure(9,figsize = (16,9))
fig9.canvas.set_window_title('Frequency - Eigenvalue Derived')
fig9.suptitle('Frequency (Eigenvalue Derived) vs. Damage level (increments of 5% undamaged weight)')
plt.plot(damagelvl,eigfreq*1000)
plt.ylim(0,100)
plt.savefig(flpath+'eigfreq'+num+dmgtype+'.png',bbox_inches='tight')
np.save(flpath+'eigfreq.npy',eigfreq*1000)
np.save(flpath+'eigfreqsd.npy',eigfreqavgsd)


#col = 1   #pick the starting column corresponding to the trial you want to look at
#flend = '_'+str(col)+'.npy'
#cell_act = np.load(flpath+'cell_act'+num+flend)
#
#for lv in np.arange(Ndamage):
#    fig12 = plt.figure(12,figsize = (16,9)) #figsize (26,12) is full screen
#    fig12.canvas.set_window_title('cell activity')  #cell outputs
#    fig12.suptitle('Cell Activity, output vs. time, damage = '+str(damagelvl[lv]))
#    for i in np.arange(Nmitral):
#        plt.subplot(Nmitral/2,2,i+1)
#        plt.plot(t,cell_act[:,i,lv])
##        plt.ylim(0,np.max(1.5*cell_act[:,:Nmitral,0]))
#        plt.ylim(0,np.max(cell_act[:,:Nmitral,:]))
#    plt.savefig(flpath+'cell_act_col'+str(col)+'_lv'+str(lv)+num+dmgtype+'.png',bbox_inches='tight')
#    plt.close()

#metadata = dict(title = 'Cell Activity',artist='Kendall Berry',comment='')
#writer = FFMpegWriter(fps=15,metadata=metadata)
#fig8 = plt.figure(8,figsize=(16,9))
##fig8.suptitle('Cell Activity, output vs. time')
#
#plt.ylim(0,np.max(yout[:,:Nmitral]))
#
#with writer.saving(fig8,"Cell_Activity_with_Damage-10_MC-One_Column.mp4",Ndamage):
#    for n in np.arange(Ndamage):
#        for i in np.arange(Nmitral):
#            plt.subplot(Nmitral/2,2,i+1)
#            plt.plot(t,yout[:,i,n])
#        writer.grab_frame()
    
#
#for n in np.arange(Ndamage):
#    plt.figure()
#    for i in np.arange(Nmitral):
#        plt.subplot(Nmitral/2,2,i+1)
#        plt.plot(t,yout[:,i,n])
#        plt.ylim(0,np.max(yout[:,:Nmitral,:]))

