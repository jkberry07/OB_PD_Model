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

                                                           
from multiprocessing import Process, Array, Lock
import numpy as np
import scipy.linalg as lin
import scipy.stats as stats
import scipy.signal as signal
from olf_bulb_fn import olf_bulb_10

Nmitral = 50

def dmg_col_50_1D(dmglvl,colnum,lock,dmgpct,d1,d2,d3,d4,pwr,IPR,freq,eigfreq,d1sd,d2sd,d3sd,\
                  d4sd,pwrsd,IPRsd,freqsd):

#INITIALIZING STUFF
    Nmitral = 50 #number of mitral cells
    Ngranule = 50 #number of granule cells     pg. 383 of Li/Hop
    Ndim = Nmitral+Ngranule #total number of cells
#    t_inh = 25 ; # time when inhalation starts
#    t_exh = 205; #time when exhalation starts
    
    Ndamagetotal = Nmitral*2 + 1  #number of damage steps
    Ndamage1 = 21
    Ncols = int(Nmitral/10)
    
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
    
    
    H0 = np.load('H0_50_58Hz.npy')
    
    H0_tot = np.sum(H0)
    
    W0 = np.load('W0_50_58Hz.npy')
    
    
    #H0 = H0 + H0*np.random.rand(np.shape(H0))
    #W0 = W0+W0*np.random.rand(np.shape(W0)) 
    
    M = 5   #average over 5 trials for each level of damage
    
    
    d1it,d2it,d3it,d4it = np.zeros(M),np.zeros(M),np.zeros(M),np.zeros(M)
    IPRit = np.zeros(M)
    frequencyit = np.zeros(M)
    pwrit = np.zeros(M)
    yout2,Sh2 = np.zeros((finalt,Ndim)),np.zeros((finalt,Ndim))
    psi = np.copy(Sh2[:,:Nmitral])
    
    for lv in np.arange(Ndamage1):
    
        damage = .05*H0[:,colnum]*lv
        Hdamaged = np.copy(H0)
        
        Hdamaged[:,colnum] = Hdamaged[:,colnum]-damage  #decrease column by 5% * dmglvl
        
        for m in np.arange(M):
            #Get the base response first
            yout,y0out,Sh,t,Omean1,Oosci1,Omeanbar1,Ooscibar1,freq0,maxlam = \
                    olf_bulb_10(Nmitral,H0,W0,P_odor1)
            yout2[:,:],y0out2,Sh2[:,:],t2,Omean2,Oosci2,Omeanbar2,\
            Ooscibar2,freq2,grow_eigs2 = olf_bulb_10(Nmitral,Hdamaged,W0,P_odor1)
            d1it[m] = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
            d2it[m] = 1-lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
            d3it[m] = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
            d4it[m] = (Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2)
            
            P_den = np.zeros((198,Ndim))
            for p in np.arange(Nmitral):
                f, P_den[:,p] = signal.periodogram(Sh2[:,p])  #periodogram returns a list of
            psi = np.zeros(Nmitral)
            for p in np.arange(Nmitral):
                psi[p] = np.sum(P_den[:,p])
            psi = psi/np.sqrt(np.sum(psi**2))
            
            IPRit[m] = 1/np.sum(psi**4)    
            pwrit[m] = np.sum(P_den)/Nmitral
            
            maxargs = np.argmax(P_den,axis=0) 
            argf = stats.mode(maxargs[maxargs!=0])
            frequencyit[m] = f[argf[0][0]]
        
    #    with lock:
        dmgpct[Ndamagetotal*colnum+lv]=np.sum(damage)/H0_tot
        IPR[Ndamagetotal*colnum+lv] = np.average(IPRit)      #Had to do 1D list, so 
        pwr[Ndamagetotal*colnum+lv] = np.average(pwrit)              #it goes column 0 damage lvl
        freq[Ndamagetotal*colnum+lv]=np.average(frequencyit)         #0,1,2,3,4...Ndamage-1, then
                                                            #col 1 damage level 0,1,2...
        IPRsd[Ndamagetotal*colnum+lv]=np.std(IPRit)
        pwrsd[Ndamagetotal*colnum+lv]=np.std(pwrit)
        freqsd[Ndamagetotal*colnum+lv]=np.std(frequencyit)
        
        d1[Ndamagetotal*colnum+lv],d1sd[Ndamagetotal*colnum+lv] = np.average(d1it),np.std(d1it)
        d2[Ndamagetotal*colnum+lv],d2sd[Ndamagetotal*colnum+lv] = np.average(d2it),np.std(d2it)
        d3[Ndamagetotal*colnum+lv],d3sd[Ndamagetotal*colnum+lv] = np.average(d3it),np.std(d3it)
        d4[Ndamagetotal*colnum+lv],d4sd[Ndamagetotal*colnum+lv] = np.average(d4it),np.std(d4it)
        
        eigfreq[Ndamagetotal*colnum+lv] = np.copy(freq2)
        
    ## Do it again for the next columns
    
    cols = np.zeros(Ncols)-1
    cols[0] = colnum
    for col in np.r_[1:Ncols]:                  #idea is to pick another column, decrease
        col2 = np.copy(colnum)                  #it to zero, then pick another
        while np.any(cols==col2):               #if col2 is the same as any previously damaged col, try a different random column
            col2 = np.random.choice(np.arange(Nmitral))
        
        dmgstep = .05*H0[:,col2]
        for lv in np.r_[1:Ndamage1]:

            Hdamaged[:,col2] = Hdamaged[:,col2]-dmgstep  #decrease column by 5% * dmglvl
            
            for m in np.arange(M):
                #Get the base response first
                yout,y0out,Sh,t,Omean1,Oosci1,Omeanbar1,Ooscibar1,freq0,maxlam = \
                        olf_bulb_10(Nmitral,H0,W0,P_odor1)
                yout2[:,:],y0out2,Sh2[:,:],t2,Omean2,Oosci2,Omeanbar2,\
                Ooscibar2,freq2,grow_eigs2 = olf_bulb_10(Nmitral,Hdamaged,W0,P_odor1)
                d1it[m] = 1-Omean1.dot(Omean2)/(lin.norm(Omean1)*lin.norm(Omean2))
                d2it[m] = 1-lin.norm(Oosci1.dot(np.conjugate(Oosci2)))/(lin.norm(Oosci1)*lin.norm(Oosci2))
                d3it[m] = (Omeanbar1 - Omeanbar2)/(Omeanbar1 + Omeanbar2)
                d4it[m] = (Ooscibar1 - Ooscibar2)/(Ooscibar1 + Ooscibar2)
                
                P_den = np.zeros((198,Ndim))
                for p in np.arange(Nmitral):
                    f, P_den[:,p] = signal.periodogram(Sh2[:,p])  #periodogram returns a list of
                psi = np.zeros(Nmitral)
                for p in np.arange(Nmitral):
                    psi[p] = np.sum(P_den[:,p])
                psi = psi/np.sqrt(np.sum(psi**2))
                
                IPRit[m] = 1/np.sum(psi**4)    
                pwrit[m] = np.sum(P_den)/Nmitral
                
                maxargs = np.argmax(P_den,axis=0) 
                argf = stats.mode(maxargs[maxargs!=0])
                frequencyit[m] = f[argf[0][0]]
            
        #    with lock:
            dmgpct[Ndamagetotal*colnum+Ndamage1*col+lv]=\
                dmgpct[Ndamagetotal*colnum+Ndamage1*col+lv-1]+np.sum(dmgstep)/H0_tot
            #for dmgpct, had to add the damgage in steps to the previous damage
            IPR[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(IPRit)      #Had to do 1D list, so 
            pwr[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(pwrit)              #it goes column 0 damage lvl
            freq[Ndamagetotal*colnum+Ndamage1*col+lv]=np.average(frequencyit)         #0,1,2,3,4...Ndamage-1, then
                                                                #col 1 damage level 0,1,2...
            IPRsd[Ndamagetotal*colnum+Ndamage1*col+lv]=np.std(IPRit)
            pwrsd[Ndamagetotal*colnum+Ndamage1*col+lv]=np.std(pwrit)
            freqsd[Ndamagetotal*colnum+Ndamage1*col+lv]=np.std(frequencyit)
            
            d1[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(d1it)
            d1sd[Ndamagetotal*colnum+Ndamage1*col+lv] = np.std(d1it)
            d2[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(d2it)
            d2sd[Ndamagetotal*colnum+Ndamage1*col+lv] = np.std(d2it)
            d3[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(d3it)
            d3sd[Ndamagetotal*colnum+Ndamage1*col+lv] = np.std(d3it)
            d4[Ndamagetotal*colnum+Ndamage1*col+lv] = np.average(d4it)
            d4sd[Ndamagetotal*colnum+Ndamage1*col+lv] = np.std(d4it)
            
            eigfreq[Ndamagetotal*colnum+Ndamage1*col+lv] = np.copy(freq2)
    
    #save all the data
   

#******************************************************************************

Ndamage = 2*Nmitral+1  #Recording 0 level damage, too, so it will be 21 levels of damage
Ncols = np.copy(Nmitral)

lvl = 0 #used to be an iterative variable, but now just a place holder
    
if __name__ == '__main__':
    dmgpct,eigfreq = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    d1, d1sd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    d2,d2sd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    d3,d3sd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    d4,d4sd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    pwr,pwrsd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    IPR,IPRsd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    freq,freqsd = Array('d',[0 for i in range(Ndamage*Ncols)]),Array('d',[0 for i in range(Ndamage*Ncols)])
    
    processes = []
    lock = Lock()
    for coln in np.arange(Nmitral):
        proc = Process(target = dmg_col_50_1D, args=(lvl,coln,lock,dmgpct,d1,\
                                                     d2,d3,d4,pwr,IPR,freq,eigfreq,\
                                                     d1sd,d2sd,d3sd,d4sd,\
                                                     pwrsd,IPRsd,freqsd))
        proc.start()
        processes.append(proc)
    
    for p in processes: p.join()

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
    d1sdfl,d2sdfl,d3sdfl,d4sdfl,IPRsdfl,pwrsdfl,frequencysdfl = "d1sd",\
    "d2sd","d3sd","d4sd","IPRsd","pwrsd","frequencysd"
    pd_type = "_50"
    np.save(d1fl+pd_type,d1),np.save(d2fl+pd_type,d2),np.save(d3fl+pd_type,d3),np.save(d4fl+pd_type,d4)
    np.save(IPRfl+pd_type,IPR),np.save(pwrfl+pd_type,pwr)
    np.save(d1sdfl+pd_type,d1sd),np.save(d2sdfl+pd_type,d2sd),np.save(d3sdfl+pd_type,d3sd),np.save(d4sdfl+pd_type,d4sd)
    np.save(IPRsdfl+pd_type,IPRsd),np.save(pwrsdfl+pd_type,pwrsd)
    np.save(eigfreqfl+pd_type,eigfreq),np.save(dmgpctfl+pd_type,dmgpct)
    np.save(frequencyfl+pd_type,freq),np.save(frequencysdfl+pd_type,freqsd)
