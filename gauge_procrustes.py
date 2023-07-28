# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:24:54 2023

@author: xusem
"""

import numpy as np
from scipy.spatial import procrustes


dim_1 = 40
dim_2 = 400
noise = 2

n = 5

iterations = 100

scores = []

for i in range(iterations):
    mat_1 = (np.random.rand(dim_1,dim_2)-0.5)*2
    mat_2 = np.random.randn(dim_1,dim_2)*noise
    mask = np.concatenate([np.zeros_like(mat_1)[:,n:], np.ones_like(mat_2)[:,:n]], axis = 1)
    aug_mat = mat_1 + mat_2*mask
    
    _,_, sim = procrustes(mat_1, aug_mat)
    scores.append(sim)

print('avg', sum(scores)/len(scores))
print('std', np.std(scores)/np.sqrt(len(scores)))