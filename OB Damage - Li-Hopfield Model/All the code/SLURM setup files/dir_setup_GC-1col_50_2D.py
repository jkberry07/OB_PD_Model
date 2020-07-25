# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os
import numpy as np

f = open('dir_setup_GC-1col_50_2D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'GC-1col_50_2D'

f.write('mkdir '+subdir1+'\n')
f.write('cp GC-1col_50_2D.py '\
        +'H0_50_2D_60Hz.npy W0_50_2D_60Hz.npy '+subdir1+'\n')


