# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os
import numpy as np

f = open('dir_setup_GC-1col_10_1D.sh','w+')



here = os.path.dirname(os.path.realpath(__file__))

subdir1 = 'GC-1col_10_1D'

f.write('mkdir '+subdir1+'\n')
f.write('cp cellout.py GC-1col_10_1D.py '\
        +'H0_original.npy W0_original.npy '+subdir1+'\n')


