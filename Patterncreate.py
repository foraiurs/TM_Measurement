#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/7/4 15:05
@author: Zhen Cheng
"""

from typing import Union, Iterable
import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import numba as nb
from code_tools import get_logger

logger = get_logger(__name__)

class Pattern(object):
    def __init__(self):
        pass

    def HadamardPattern(self, n_had=16):
        """
        Creat 3d hadamard pattern of size (n**2)*n*n
        """

        N_had = n_had**2
        H_had = hadamard(N_had)
        B_had = np.reshape(H_had, (N_had, n_had, n_had))
        B_had = np.array(B_had, dtype=np.int8)
        return B_had

    # @nb.jit(nopython=True, parallel=True)
    def LeePattern(
        self,
        phase: Union[np.ndarray, Iterable],
        amplitude: Union[np.ndarray, Iterable, float, int],
        alpha=0.05,
        rot=7 * np.pi / 4,
    ):
        """
        amplitude is either a constant or the array shape = phase_4.shape

        Lee method for generating amplitude pattern that simulates phase_4
        t(x,y) = 0.5 * [1+cos(2*pi*alpha*c(x,y) - phase_4(x,y))
        c(x,y) = cos(rot)*x + sin(rot)*y
        h(x,y)  = 1 for t(x,y) > 0.5 , 0 otherwise
        alpha: carrier frequency, providing a large enough separation of the 1st order
        from the 0th order beam
        rot: adjust angle
        """

        DR = 1 - amplitude

        if phase.ndim == 2:
            x_lee, y_lee = np.meshgrid(
                np.arange(phase.shape[1]), np.arange(phase.shape[0])
            )
            # print("x_lee", x_lee)
            # print("y_lee", y_lee)
            # c_val = x_lee - y_lee
            c_val = np.cos(rot) * x_lee + np.sin(rot) * y_lee
            # print("c_val", c_val)

        elif phase.ndim == 3:
            x_lee, y_lee = np.meshgrid(
                np.arange(phase.shape[2]), np.arange(phase.shape[1])
            )
            # c_val = x_lee - y_lee
            c_val = np.cos(rot) * x_lee + np.sin(rot) * y_lee
            # c_val = np.expand_dims(c_val2, axis=0)
            c_val = c_val.reshape((1, phase.shape[1], phase.shape[2]))
            # c_val = np.repeat_mea(c_val, self.phase_array.shape[0], axis=0)
            # print("c_val", c_val)
        else:
            logger.error("Invalid dimension of phase array. Only 2D or 3D arrays are supported.")
            raise ValueError(
                "Invalid dimension of phase array. Only 2D or 3D arrays are supported."
            )


        phase = np.cos(2 * np.pi * alpha * c_val - phase)
        # logger.info("phase_array dtype:{}".format(phase.dtype))
        phase = np.where(phase > DR, 1, 0)
        phase = np.array(phase, dtype=np.uint8)
        # logger.info("phase_array_binary dtype:{}".format(phase.dtype))
        return phase

    # @nb.jit(nopython=True, parallel=True)
    def BoydPattern(
        self,
        phase: Union[np.ndarray, Iterable],
        amplitude: Union[np.ndarray, Iterable, float, int],
        x0=20,
        rot=7 * np.pi / 4,
    ):
        """
        Robert W. Boyd, Rapid generation of light beams carrying orbital angular momentum
        alpha=1/x0
        amplitude is either a constant or the array shape = phase_4.shape

        """

        w = np.arcsin(amplitude)

        if phase.ndim == 2:
            x_lee, y_lee = np.meshgrid(
                np.arange(phase.shape[1]), np.arange(phase.shape[0])
            )
            # c_val = x_lee - y_lee
            c_val = np.cos(rot) * x_lee + np.sin(rot) * y_lee

        elif phase.ndim == 3:
            x_lee, y_lee = np.meshgrid(
                np.arange(phase.shape[2]), np.arange(phase.shape[1])
            )
            # c_val = x_lee - y_lee
            c_val = np.cos(rot) * x_lee + np.sin(rot) * y_lee
            # c_val = np.expand_dims(c_val2, axis=0)
            c_val = c_val.reshape((1, phase.shape[1], phase.shape[2]))
            # c_val = np.repeat_mea(c_val, self.phase_array.shape[0], axis=0)
        else:
            logger.error("Invalid dimension of phase array. Only 2D or 3D arrays are supported.")
            raise ValueError(
                "Invalid dimension of phase array. Only 2D or 3D arrays are supported."
            )

        pattern = np.cos(2 * np.pi * c_val / x0 - phase) - np.cos(w)
        # logger.info("pattern dtype:{}".format(pattern.dtype))
        pattern = np.where(pattern > 0, 1, 0)
        pattern = np.array(pattern, dtype=np.uint8)
        # logger.info("pattern_binary dtype:{}".format(pattern.dtype))
        return pattern


if __name__ == "__main__":
    # 假设给定一个示例数组
    alpha = 0.05
    rot = 7 * np.pi / 4
    amplitude1 = 1
    x = np.full((100, 100), np.pi / 2)
    # x[150:199, 150:199] = np.pi / 4
    # x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # rot = [0, np.pi/4, np.pi/2, (3/4)*np.pi, np.pi, (5/4)*np.pi, (3/2)*np.pi, (7/4)*np.pi, 2*np.pi]
    pat = Pattern()
    h = pat.LeePattern(x, amplitude1, alpha, rot)
    print("h(x,y):{}".format(h.shape))

    plt.figure(1)
    # 绘制h_array的热图
    plt.imshow(h, cmap="gray", origin="lower", extent=(0, h.shape[1], 0, h.shape[0]))
    plt.colorbar(label="h(x,y)")
    plt.title("h_array Heatmap")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)

    # hadamard
    n_test = 16
    hadamardpat = pat.HadamardPattern(n_test)
    phaseBasis = np.where(hadamardpat == 1, 0, np.pi)
    print("hadamard pattern:{} ,dtype:{}".format(hadamardpat.shape, hadamardpat.dtype))
    i_had = 100

    plt.figure(2)
    plt.imshow(hadamardpat[i_had, :, :], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("pattern{}".format(i_had))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)

    # 加载的lee全息图 超像素大小10*10
    x_superpixel = 70
    leebasis = np.kron(phaseBasis, np.ones((x_superpixel, x_superpixel)))
    print("leebasis.shape:{}".format(leebasis.shape))
    amplitude2 = np.full((leebasis.shape[1], leebasis.shape[2]), 0.5)
    amplitude2 = amplitude2.reshape((1, amplitude2.shape[0], amplitude2.shape[1]))
    h2 = pat.LeePattern(leebasis, amplitude2, alpha, rot)
    print("h2:{}, h2.shape:{}, dtype:{}".format(h2[5], h2.shape, h2.dtype))

    # 加载二维
    dpinum = 80
    plt.figure(
        num=3,
        figsize=(x_superpixel * n_test / dpinum, x_superpixel * n_test / dpinum),
        dpi=dpinum,
    )
    leebasis_ = np.kron(phaseBasis[i_had, :, :], np.ones((x_superpixel, x_superpixel)))
    amplitude3 = np.full_like(leebasis_, 1)
    h2_ = pat.LeePattern(leebasis_, amplitude3, alpha, rot)
    plt.imshow(h2_, cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("leehol_twodinm{}".format(i_had))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)

    # 加载三维
    plt.figure(
        num=4,
        figsize=(x_superpixel * n_test / dpinum, x_superpixel * n_test / dpinum),
        dpi=dpinum,
    )
    plt.imshow(h2[i_had, :, :], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("leehol{}".format(i_had))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)

    # 加载Boyd光栅

    # amplitude4 = 0.5
    x0 = 20
    amplitude4 = np.full((leebasis.shape[1], leebasis.shape[2]), 0.5)
    amplitude4 = amplitude4.reshape((1, amplitude4.shape[0], amplitude4.shape[1]))
    h3 = pat.BoydPattern(leebasis, amplitude4, x0, rot)
    print("h3:{}, h3.shape:{}, dtype:{}".format(h3[5], h3.shape, h3.dtype))
    plt.figure(
        num=5,
        figsize=(x_superpixel * n_test / dpinum, x_superpixel * n_test / dpinum),
        dpi=dpinum,
    )
    plt.imshow(h3[i_had, :, :], cmap="viridis", origin="lower")
    plt.colorbar()
    plt.title("boydhol{}".format(i_had))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)
    plt.savefig('example.svg')

    plt.show()

