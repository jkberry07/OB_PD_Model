# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os


f = open('dir_setup_MC-seed_10_2D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'MC-seed_10_2D'

f.write('mkdir '+subdir1+'\n')
f.write('cp MC-seed_10_2D.py '\
        +'H0_10_2D_65Hz.npy W0_10_2D_65Hz.npy '+subdir1+'\n')


