# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:09:06 2019

@author: wmmjk
"""

import os




here = os.path.dirname(os.path.realpath(__file__))

submitname = 'submit_PD-flat_20_1D.sh'

f3 = open(submitname,'w+')

subdir1 = 'PD-flat_20_1D'

f3.write('cd '+subdir1+'\n')



jobname = 'PD-flat_20_1D.sh'
f2 = open(os.path.join(here,subdir1,jobname),'w+')
f2.write('#!/bin/bash -l\n')
f2.write('#SBATCH -J PD_20_1D_%A\n')
f2.write('#SBATCH -o PD_20_1D_%A.txt\n')
f2.write('#SBATCH -e PD_20_1De_%A.txt\n')
f2.write('#SBATCH -t 2-00:00:00\n')
f2.write('#SBATCH --mail-type=all\n')
f2.write('#SBATCH --mail-user=jkberry@ucdavis.edu\n')
f2.write('#SBATCH --array=1\n') 
f2.write('module load bio3\n')
f2.write('srun python PD-flat_20_1D.py')

f3.write('chmod u+x '+jobname+'\n')
f3.write('sbatch -p med2 '+jobname+'\n')
f3.write('cd ..\n')
    



