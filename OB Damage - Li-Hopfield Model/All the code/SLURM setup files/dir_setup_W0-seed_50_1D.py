# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os


f = open('dir_setup_W0-seed_50_1D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'W0-seed_50_1D'

f.write('mkdir '+subdir1+'\n')
f.write('cp W0-seed_50_1D.py '\
        +'H0_50_53Hz.npy W0_50_53Hz.npy '+subdir1+'\n')


