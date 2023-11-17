#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/5/14 14:38
@author: Zhen Cheng
"""

import numpy as np
from ALP4 import *
import time
from Patterncreate import *

# Load the Vialux .dll
with ALP4(version="4.3") as DMD:
    # Initialize the device
    DMD.Initialize()
    print(DMD.nSizeY)
    print(DMD.nSizeX)

    DMD.DevControl(
        controlType=ALP_TRIGGER_EDGE, value=ALP_EDGE_RISING
    )  # ALP_EDGE_RISING

    dmdtype = DMD.DevInquire(inquireType=ALP_DEV_DMDTYPE)
    print(
        "dmdtype:{}".format(dmdtype)
    )  # ALP_DMDTYPE_1080P_095A / DLP9500 / 1920x1080 pixels
    triggerEdge = DMD.DevInquire(inquireType=ALP_TRIGGER_EDGE)
    print("triggerEdge:{}".format(triggerEdge))  # ALP_EDGE_RISING
    memory = DMD.DevInquire(inquireType=ALP_AVAIL_MEMORY)
    print("memory:{}".format(memory))  # ALP_EDGE_RISING
    DDCtemperature = DMD.DevInquire(inquireType=ALP_DDC_FPGA_TEMPERATURE)
    print(
        "DDCtemperature:{}".format(DDCtemperature)
    )  # DLPC FPGA temperature (The value is written as 1/256 °C)
    APPStemperature = DMD.DevInquire(inquireType=ALP_APPS_FPGA_TEMPERATURE)
    print("APPStemperature:{}".format(APPStemperature))  # Applications FPGA
    PCBtemperature = DMD.DevInquire(inquireType=ALP_PCB_TEMPERATURE)
    print(
        "PCBtemperature:{}".format(PCBtemperature)
    )  # the internal temperature of the temperature sensor IC

    effectiveDMDsize = min(DMD.nSizeY, DMD.nSizeX)
    DMDextraleft = round((DMD.nSizeX - effectiveDMDsize) / 2)
    DMDextraright = DMD.nSizeX - effectiveDMDsize - DMDextraleft

    # Binary amplitude image (0 or 1)
    bitDepth = 1
    # imgBlack = np.zeros([512,DMD.nSizeX])
    # imgWhite = np.ones([512,DMD.nSizeX])*(2**8-1)
    # im = np.round(np.kron(np.random.rand(32,24),np.ones([32,32]))*255)
    # imgSeq = np.concatenate([imgBlack.ravel(),imgWhite.ravel()])
    # imgproject = np.zeros((DMD.nSizeY,DMD.nSizeX))
    imgphase = np.full((DMD.nSizeY, DMD.nSizeY), np.pi / 2, dtype=np.uint8)
    imgphase = np.pad(
        imgphase, ((0, 0), (DMDextraleft, DMDextraright)), "constant", constant_values=0
    )
    leeimg = Pattern()
    array1 = leeimg.LeePattern(imgphase, 1, 0.05, 0) * 255
    array2 = leeimg.LeePattern(imgphase, 1, 0.05, 7 * np.pi / 4) * 255
    array = np.stack((array1, array2), axis=0)
    # array1 = np.array(array1*255, dtype=np.uint8)
    # array2 = np.array(array2*255, dtype=np.uint8)
    # imgbinary = np.concatenate([array1.ravel(), array2.ravel()])

    # plt.figure(1)
    # plt.imshow(imgbinary, cmap='viridis', origin='lower')
    # plt.colorbar(label='(x,y)')
    # plt.title('Lee')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.grid(0)

    # Allocate the onboard memory for the image sequence
    DMD.SeqAlloc(nbImg=2, bitDepth=bitDepth)

    DMD.SeqControl(ALP_BITNUM, 1)  # <bit number> 1 … BitPlanes
    DMD.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED)  # Operation without dark phase_4
    DMD.SeqControl(ALP_SEQ_REPEAT, 1)  # the sequence is displayed this number of times

    DMD.SetTiming(
        illuminationTime=1000000, pictureTime=1000000, synchPulseWidth=100
    )  # μs

    BITPLANES = DMD.SeqInquire(ALP_BITPLANES)
    print("BITPLANES:{}".format(BITPLANES))
    BITNUM = DMD.SeqInquire(ALP_BITNUM)
    print("BITNUM:{}".format(BITNUM))
    BIN_MODE = DMD.SeqInquire(ALP_BIN_MODE)
    print("BIN_MODE:{}".format(BIN_MODE))
    PICNUM = DMD.SeqInquire(ALP_PICNUM)
    print("PICNUM:{}".format(PICNUM))
    MIN_PICTURE_TIME = DMD.SeqInquire(ALP_MIN_PICTURE_TIME)
    print("MIN_PICTURE_TIME:{}".format(MIN_PICTURE_TIME))
    MIN_ILLUMINATE_TIME = DMD.SeqInquire(ALP_MIN_ILLUMINATE_TIME)
    print("MIN_ILLUMINATE_TIME:{}".format(MIN_ILLUMINATE_TIME))
    ON_TIME = DMD.SeqInquire(ALP_ON_TIME)
    print("ON_TIME:{}".format(ON_TIME))
    SCROLL_FROM_ROW = DMD.SeqInquire(ALP_SCROLL_FROM_ROW)
    print("SCROLL_FROM_ROW:{}".format(SCROLL_FROM_ROW))

    print("Start put")
    DMD.SeqPut(imgData=array1, PicOffset=0, PicLoad=1)
    DMD.SeqPut(imgData=array2, PicOffset=1, PicLoad=1)

    print("Start run")
    DMD.Run(loop=False)

    # plt.show()
    DMD.Wait()

    DMD.Halt()
    DMD.FreeSeq()

    DMD.SeqAlloc(nbImg=1, bitDepth=bitDepth)

    DMD.SeqControl(ALP_BITNUM, 1)  # <bit number> 1 … BitPlanes
    DMD.SeqControl(ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED)  # Operation without dark phase_4
    DMD.SeqControl(ALP_SEQ_REPEAT, 1)  # the sequence is displayed this number of times

    DMD.SetTiming(
        illuminationTime=2000000, pictureTime=2000000, synchPulseWidth=100
    )  # μs

    print("Start put 2")
    DMD.SeqPut(imgData=array1, PicOffset=0, PicLoad=1)

    print("Start run 2")
    DMD.Run(loop=False)

    # plt.show()
    DMD.Wait()




    # time.sleep(1000000)

# DMD.Halt()
# # Free the sequence from the onboard memory
# DMD.FreeSeq()
# # De-allocate the device
# DMD.Free()
