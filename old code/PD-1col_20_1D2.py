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

                                                           
from multiprocessing import Pool
import numpy as np
import scipy.linalg as lin
import scipy.stats as stats
import scipy.signal as signal
from olf_bulb_fn import olf_bulb_10
import time

tm1 = time.time()

Nmitral0 = 20  #define network size in parent process

def dmg_col_20_1D(colnum):

#INITIALIZING STUFF
    Nmitral = 20
    Ngranule = np.copy(Nmitral) #number of granule cells     pg. 383 of Li/Hop
    Ndim = Nmitral+Ngranule #total number of cells
#    t_inh = 25 ; # time when inhalation starts
#    t_exh = 205; #time when exhalation starts
    
#    Ndamagetotal = Nmitral*2 + 1  #number of damage steps
    Ndamage = 21
    Ncols = int(Nmitral/10)     #define number of columns to damage
    
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
    
    
    H0 = np.load('H0_20_52Hz.npy')   #load weight matrix
    
    H0_tot = np.sum(H0)         #get sum of the weights
    
    W0 = np.load('W0_20_52Hz.npy')  #load weight matrix
    
    
    #H0 = H0 + H0*np.random.rand(np.shape(H0))
    #W0 = W0+W0*np.random.rand(np.shape(W0)) 
    
    M = 5   #average over 5 trials for each level of damage
    
    
    #initialize iterative variables
    d1it,d2it,d3it,d4it = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
    IPRit = np.zeros(M)
    frequencyit = np.zeros(M)
    pwrit = np.zeros(M)
    yout2,Sh2 = np.zeros((finalt,Ndim)),np.zeros((finalt,Ndim))
    psi = np.copy(Sh2[:,:Nmitral])
    
    #initialize quantities to be returned at end of the process
    dmgpct1 = np.zeros(Ndamage)
    eigfreq1 = np.zeros(Ndamage)
    d11 = np.zeros(Ndamage)
    d21 = np.zeros(Ndamage)
    d31 = np.zeros(Ndamage)
    d41 = np.zeros(Ndamage)
    pwr1 = np.zeros(Ndamage)
    IPR1 = np.zeros(Ndamage)
    freq1 = np.zeros(Ndamage)
    
    cols = [i for i in range(Ncols)]    #initialize as a list so that it can be integers
    for i in np.arange(Ncols):
        cols[i] = int(np.mod(colnum+i,Nmitral))   #Define which columns to damage
    
    for lv in np.arange(Ndamage):
    
        damage = .05*H0[:,cols]*lv    #define amount of damage given the level 
        Hdamaged = np.copy(H0)      #reinitialize the damaged matrix
        
        Hdamaged[:,cols] = Hdamaged[:,cols]-damage  #decrease columns by 5% * dmglvl
        for m in np.arange(M):
            #Get the base response first
            yout,y0out,Sh,t,Omean1,Oosci1,Omeanbar1,Ooscibar1,freq0,maxlam = \
                    olf_bulb_10(Nmitral,H0,W0,P_odor1)  #calls function olf_bulb_10
            #Then get respons of damaged network
            yout2[:,:],y0out2,Sh2[:,:],t2,Omean2,Oosci2,Omeanbar2,\
            Ooscibar2,freq2,grow_eigs2 = olf_bulb_10(Nmitral,Hdamaged,W0,P_odor1)
            #calculate distance measures
            d1it[m] = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
            d2it[m] = 1-lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
            d3it[m] = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
            d4it[m] = np.real((Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2))             
            
            #calculate spectral density and "wave function" to get average power and IPR
            P_den = np.zeros((198,Ndim))
            for p in np.arange(Nmitral):
                f, P_den[:,p] = signal.periodogram(Sh2[:,p])  #periodogram returns a list of
            psi = np.zeros(Nmitral)
            for p in np.arange(Nmitral):
                psi[p] = np.sum(P_den[:,p])
            psi = psi/np.sqrt(np.sum(psi**2))
            
            IPRit[m] = 1/np.sum(psi**4)    
            pwrit[m] = np.sum(P_den)/Nmitral
            
            #get the frequency according to the adiabatic analysis
            maxargs = np.argmax(P_den,axis=0) 
            argf = stats.mode(maxargs[maxargs!=0])
            frequencyit[m] = f[argf[0][0]]
#            print(cols)
#            print(time.time()-tm1)
#        
#        print('level',lv)
        #Get the returned variables for each level of damage
        dmgpct1[lv]=np.sum(damage)/H0_tot
        IPR1[lv] = np.average(IPRit)      #Had to do 1D list, so 
        pwr1[lv] = np.average(pwrit)              #it goes column 0 damage lvl
        freq1[lv]=np.average(frequencyit)         #0,1,2,3,4...Ndamage-1, then
                                                            #col 1 damage level 0,1,2...
#        IPRsd[lv]=np.std(IPRit)
#        pwrsd[lv]=np.std(pwrit)
#        freqsd[lv]=np.std(frequencyit)
        
        d11[lv]= np.average(d1it)
        d21[lv] = np.average(d2it)
        d31[lv] = np.average(d3it)
        d41[lv] = np.average(d4it)
#        d1sd[lv] =  np.std(d1it)
#        d2sd[lv] = np.std(d2it)
#        d3sd[lv]=np.std(d3it)
#        d4sd[lv]=np.std(d4it)
        
        eigfreq1[lv] = np.copy(freq2)
        
   
    return dmgpct1,eigfreq1,d11,d21,d31,d41,pwr1,IPR1,freq1
    
    
    #save all the data
   

#******************************************************************************

Ndamage0 = 21  #Recording 0 level damage, too, so it will be 21 levels of damage

lvl = 0 #used to be an iterative variable, but now just a place holder
    
if __name__ == '__main__':
    
    dmgpct = np.zeros((Ndamage0,Nmitral0))
    eigfreq = np.zeros((Ndamage0,Nmitral0))
    d1 = np.zeros((Ndamage0,Nmitral0))
    d2 = np.zeros((Ndamage0,Nmitral0))
    d3 = np.zeros((Ndamage0,Nmitral0))
    d4 = np.zeros((Ndamage0,Nmitral0))
    pwr = np.zeros((Ndamage0,Nmitral0))
    IPR = np.zeros((Ndamage0,Nmitral0))
    freq = np.zeros((Ndamage0,Nmitral0))
    
    for cols in range(Nmitral0):
        dmgpct[:,cols],eigfreq[:,cols],d1[:,cols],d2[:,cols],d3[:,cols],d4[:,cols],\
        pwr[:,cols],IPR[:,cols],freq[:,cols] = dmg_col_20_1D(cols)
    
    
#    dmgpct = [output[i].get()[0] for i in np.arange(Ncolumns)]
#    eigfreq = [output[i].get()[1] for i in np.arange(Ncolumns)]
#    d1 = [output[i].get()[2] for i in np.arange(Ncolumns)]
#    d2 = [output[i].get()[3] for i in np.arange(Ncolumns)]
#    d3 = [output[i].get()[4] for i in np.arange(Ncolumns)]
#    d4 = [output[i].get()[5] for i in np.arange(Ncolumns)]
#    pwr = [output[i].get()[6] for i in np.arange(Ncolumns)]
#    IPR = [output[i].get()[7] for i in np.arange(Ncolumns)]
#    freq = [output[i].get()[8] for i in np.arange(Ncolumns)]
    
#    output = dmg_col_20_1D(Nmitral,Ndamage,0)
    
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
    pd_type = "_20"
    np.save(d1fl+pd_type,d1),np.save(d2fl+pd_type,d2),np.save(d3fl+pd_type,d3),np.save(d4fl+pd_type,d4)
    np.save(IPRfl+pd_type,IPR),np.save(pwrfl+pd_type,pwr)
#    np.save(d1sdfl+pd_type,d1sd),np.save(d2sdfl+pd_type,d2sd),np.save(d3sdfl+pd_type,d3sd),np.save(d4sdfl+pd_type,d4sd)
#    np.save(IPRsdfl+pd_type,IPRsd),np.save(pwrsdfl+pd_type,pwrsd)
    np.save(eigfreqfl+pd_type,eigfreq),np.save(dmgpctfl+pd_type,dmgpct)
    np.save(frequencyfl+pd_type,freq)
#    np.save(frequencysdfl+pd_type,freqsd)
