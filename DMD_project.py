#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/7/29 13:08
@author: Zhen Cheng
"""
from ALP4 import *
from DMD_holography import *
import code_tools


logger = code_tools.get_logger("DMD_project")

logger.debug('DMD_project debug mode ON.')

# Set parameters
load = False
axial_method = 2  # 1: coaxial 2: offaxial
holo_method = 1  # 1: LeePattern, 2: BoydPattern
bitDepth = 1
hadamardSize = 32  # (16. 32. 64)
phase_4 = (0, np.pi / 2, np.pi, 3 * np.pi / 2)
amplitude = 1
patnumber = np.size(phase_4) * (hadamardSize ** 2)
illuminationTime = 10000  # μs (Minimum time = 7400μs, )
pictureTime = illuminationTime  # μs
repeat_mea = 100000

# Initialize the DMD
with ALP4(version="4.3") as DMD:
    # Initialize the DMD
    DMD.Initialize()
    DMDheight = DMD.nSizeY
    DMDwidth = DMD.nSizeX

    DMD.SeqAlloc(nbImg=patnumber, bitDepth=bitDepth)

    DMD.SeqControl(ALP_BITNUM, 1)  # <bit number> 1 … BitPlanes
    DMD.SeqControl(
        ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED
    )  # Operation without dark phase_4
    DMD.SeqControl(
        ALP_SEQ_REPEAT, repeat_mea
    )  # the sequence is displayed this number of times
    # DMD.SeqControl(ALP_FIRSTFRAME, 0)
    # DMD.SeqControl(ALP_LASTFRAME, 0)

    DMD.SetTiming(
        illuminationTime=illuminationTime,
        pictureTime=pictureTime,
        synchPulseWidth=100,
    )  # μs

    if load:
        block_size_loop_put = 1024
        for i_put in range(0, hadamardSize ** 2, block_size_loop_put):
            logger.info("------------loading put project number {}-----------".format(i_put))
            start_put = i_put
            end_put = min(i_put + block_size_loop_put, hadamardSize ** 2)
            if axial_method == 1:
                file_name_proj = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_coaxial_{start_put}_{end_put}_methold{holo_method}.npy'
            elif axial_method == 2:
                file_name_proj = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_offaxial_{start_put}_{end_put}_methold{holo_method}.npy'
            else:
                raise ValueError("Must choose axial_methold 1: coaxial or 2: offaxial")
            current_phase_array = np.load('./measure_patterns/' + file_name_proj)
            DMD.SeqPut(imgData=current_phase_array, PicOffset=np.size(phase_4) * i_put,
                       PicLoad=current_phase_array.shape[0])

    logger.info("Start projection")
    DMD.Run(loop=False)
    DMD.Wait()
