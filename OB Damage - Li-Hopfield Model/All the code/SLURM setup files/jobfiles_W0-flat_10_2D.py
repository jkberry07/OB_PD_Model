# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os




here = os.path.dirname(os.path.realpath(__file__))

submitname = 'submit_W0-flat_10_2D.sh'

f3 = open(submitname,'w+')

subdir1 = 'W0-flat_10_2D'

f3.write('cd '+subdir1+'\n')



jobname = 'W0-flat_10_2D.sh'
f2 = open(os.path.join(here,subdir1,jobname),'w+')
f2.write('#!/bin/bash -l\n')
f2.write('#SBATCH -J W0_10_2D_%A\n')
f2.write('#SBATCH -o W0_10_2D_%A.txt\n')
f2.write('#SBATCH -e W0_10_2De_%A.txt\n')
f2.write('#SBATCH -t 2-00:00:00\n')        
f2.write('#SBATCH --mail-type=all\n')
f2.write('#SBATCH --mail-user=jkberry@ucdavis.edu\n')
f2.write('#SBATCH --array=1\n')
f2.write('module load bio3\n')
f2.write('srun python W0-flat_10_2D.py')

f3.write('chmod u+x '+jobname+'\n')
f3.write('sbatch -p med2 '+jobname+'\n')
f3.write('cd ..\n')


