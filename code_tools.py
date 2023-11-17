#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/8/5 17:08
@author: Zhen Cheng
"""

import logging
import sys
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch


def get_logger(name):
        loglevel = logging.DEBUG
        logger = logging.getLogger(name)
        if not getattr(logger, 'handler_set', None):
            logFormatter = logging.Formatter("%(asctime)s %(name)-15.15s %(levelname)-8.8s %(message)s") #[%(threadName)-12.12s]
            fileHandler = logging.FileHandler("{}/{}.log".format('./', 'MMF_tm'))
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
            streamHandler = logging.StreamHandler(sys.stdout)
            streamHandler.setFormatter(logFormatter)
            logger.addHandler(streamHandler)
            logger.setLevel(loglevel)
            logger.handler_set = True
        return logger

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images. imgs必须是三维数组"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            img = img.numpy()
        # PIL图片
        im = ax.imshow(img)
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    fig.colorbar(im, ax=axes)
    return axes

def plot_lines(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器。"""
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和。"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()

if __name__ == '__main__':
    hadamardSize = 64
    camera_length_afterDownSample = 200
    file_name_tm = f'MMF_{hadamardSize}hadamard_{camera_length_afterDownSample}camera_img_size.npy'
    tm_array = np.load('./measure_tm/' + file_name_tm)
    TM_phase_array = np.angle(tm_array)
    TM_amplitude_array = np.abs(tm_array)
    show_images(TM_amplitude_array.reshape((1,) + TM_amplitude_array.shape), 1, 1, scale=30)
    plt.savefig('TM_amplitude_array.svg', format='svg')
    show_images(TM_phase_array.reshape((1,) + TM_phase_array.shape), 1, 1, scale=30)
    plt.savefig('TM_phase_array.svg', format='svg')
    plt.show()