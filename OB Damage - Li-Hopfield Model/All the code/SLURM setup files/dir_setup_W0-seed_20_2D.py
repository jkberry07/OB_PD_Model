# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os

f = open('dir_setup_W0-seed_20_2D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'W0-seed_20_2D'

f.write('mkdir '+subdir1+'\n')
f.write('cp W0-seed_20_2D.py '\
        +'H0_20_2D_50Hz.npy W0_20_2D_50Hz.npy '+subdir1+'\n')


