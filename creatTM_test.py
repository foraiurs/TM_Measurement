#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/7/20 15:06
@author: Zhen Cheng
"""

from DMD_holography import *

# 验证时需要把createTM_cam中的block_size_loop_tm设置为 1

complex_array = np.zeros((16**2, 16), dtype=np.complex128)


input_array = np.random.random((16*4, 16, 16))


reshaped_array = input_array.reshape(16*4, 16**2)
print("reshaped_array", reshaped_array)


indexes = np.arange(0, 16*4, 4)
print("indexes", indexes)


real_part = (reshaped_array[indexes] - reshaped_array[indexes + 2]) / 4
print("real_part", real_part, real_part.shape)
imag_part = (reshaped_array[indexes + 3] - reshaped_array[indexes + 1]) / 4

# 将计算得到的实部和虚部组成复数并赋值到对应的位置
complex_array[:, 0:16] = (real_part + 1j * imag_part).T
complex_array = np.dot(complex_array, hadamard(16))
print("complex_array", complex_array, complex_array.shape)


tmmea = ImgAcquire()
TM = tmmea.createTM_cam(input_array, 16)
print("TM", TM, TM.shape)
xiangtong = TM-complex_array
print(np.max(xiangtong), np.min(xiangtong))
