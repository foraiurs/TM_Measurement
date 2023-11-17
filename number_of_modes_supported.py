#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/8/14 12:42
@author: Zhen Cheng
"""
import numpy as np

NA = np.array([0.15, 0.22, 0.22, 0.29])
a = np.array([105, 50, 50, 100]) # μm
lamda = np.array([0.6328, 0.6328, 1.5, 0.6328]) # μm
V = 2 * np.pi * (a/2) * NA / lamda
pola = 1
M = np.ceil(V**2/2.*pola/2.).astype(int) # the number of mmf modes
length = np.array([100, 30, 200, 150]) # cm
print('实验使用MMF长度为{}cm，支持的模式数量:{}'.format(length[0], M[0]))
print('Compressively sampling MMF长度为{}cm，支持的模式数量:{}'.format(length[1], M[1]))
print('Polarization control MMF长度为{}cm，支持的模式数量:{}'.format(length[2], M[2]))