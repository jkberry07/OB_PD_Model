# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os

f = open('dir_setup_OI-flat_20_1D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'OI-flat_20_1D'

f.write('mkdir '+subdir1+'\n')
f.write('cp OI-flat_20_1D.py '\
        +'H0_20_52Hz.npy W0_20_52Hz.npy '+subdir1+'\n')


