#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/8/3 16:52
@author: Zhen Cheng
"""

import numpy as np
from ALP4 import *
import time
from Patterncreate import *
from DMD_holography import *
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
import matplotlib.pyplot as plt
import code_tools

logger = code_tools.get_logger("MMF_tm_focus")

logger.debug('MMF_tm_focus debug mode ON.')

# Set parameters
load = False
axial_method = 1 # 1: coaxial 2: offaxial
holo_method = 1  # 1: LeePattern, 2: BoydPattern
bitDepth = 1
hadamardSize = 64  # (16. 32. 64)
if axial_method == 1:
    referenceFraction = 0.35  # coaxial:0.35, offaxial:0
elif axial_method == 2:
    referenceFraction = 0  # coaxial:0.35, offaxial:0
else:
    logger.error("Must choose axial_methold 1: coaxial or 2: offaxial")
    raise ValueError("Must choose axial_methold 1: coaxial or 2: offaxial")
phase_4 = (0, np.pi / 2, np.pi, 3 * np.pi / 2)
amplitude = 1
alpha = 0.1  # lee, the spacing between grating stripes is 1/alpha pixels
x0 = 10  # boyd
rot = 7 * np.pi / 4
patnumber = np.size(phase_4) * (hadamardSize ** 2)
illuminationTime = 15000  # μs (Minimum time = 7400μs, )
pictureTime = illuminationTime  # μs
illuminationTime_focus = 500000
pictureTime_focus = illuminationTime_focus
repeat_mea = 1
repeat_focus = 100
# camera_roi = (0, 0, 1279, 1023)
# roi (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
camera_roi = (576, 386, 726, 536)  # The set ROI is different from the actual ROI of the camera and needs to be cropped
camera_length = 150  # The cropped pattern after camera acquisition
camera_length_afterDownSample = 128 # camera_length_afterDownSample <= camera_length (16.32.64.128)
camimg = ImgAcquire()

# Initialize the camera
try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from cmos_examples import configure_path

    configure_path()
except ImportError:
    configure_path = None

with ALP4(version="4.3") as DMD:

    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()

        if len(available_cameras) < 1:
            logger.error("no cameras detected")

        with sdk.open_camera(available_cameras[0]) as camera:
            camera.exposure_time_us = 100  # set exposure time(μs)
            camera.roi = camera_roi  # set roi
            cam_leftuppery, cam_leftupperx = camimg.getparameter_cam(camera_roi, camera_length)
            camera.operation_mode = 1  # hardware trigger
            camera.frames_per_trigger_zero_for_unlimited = 1  # The number of frames generated per software or hardware trigger
            camera.trigger_polarity = 0  # rising-edge trigger，falling=1
            camera.image_poll_timeout_ms = 2000  # polling timeout(ms)
            # camera.issue_software_trigger()
            camera.arm(2)

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

            DMD.SetTiming(
                illuminationTime=illuminationTime,
                pictureTime=pictureTime,
                synchPulseWidth=100,
            )  # μs

            dmdpat = PatternProject(DMDwidth, DMDheight)

            superpixel = dmdpat.getparameter(hadamardSize, referenceFraction)

            if load:
                block_size_loop_put = 1024
                for i_put in range(0, hadamardSize**2, block_size_loop_put):
                    logger.info("------------loading put project number {}-----------".format(i_put))
                    start_put = i_put
                    end_put = min(i_put + block_size_loop_put, hadamardSize**2)
                    if axial_method == 1:
                        file_name_proj = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_coaxial_{start_put}_{end_put}_methold{holo_method}.npy'
                    elif axial_method == 2:
                        file_name_proj = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_offaxial_{start_put}_{end_put}_methold{holo_method}.npy'
                    else:
                        logger.error("Must choose axial_methold 1: coaxial or 2: offaxial")
                        raise ValueError("Must choose axial_methold 1: coaxial or 2: offaxial")
                    current_phase_array = np.load('./measure_patterns/' + file_name_proj)
                    DMD.SeqPut(imgData=current_phase_array, PicOffset=np.size(phase_4)*i_put,
                               PicLoad=current_phase_array.shape[0])

            camera_img = np.zeros((patnumber, camera_length, camera_length), dtype=np.uint16)
            i_get_img_number = 0

            logger.info("Start projection")
            DMD.Run(loop=False)

            logger.info("Camera start capture")
            for i_cam in range(patnumber):
                frame = camera.get_pending_frame_or_null()
                if frame is not None:
                    # 相机采集图像数值是10bits，所以是0-1023的整数
                    camera_img[i_cam] = frame.image_buffer[cam_leftuppery:cam_leftuppery + camera_length,
                                    cam_leftupperx:cam_leftupperx + camera_length]
                    i_get_img_number += 1
                    if frame.frame_count != i_get_img_number:
                        logger.error("frame and img_number do not match. Camera capture error.")
                        raise ValueError("frame and img_number do not match. Camera capture error. Please run the "
                                         "program again.")
                    # if i_get_img_number % 4000 == 0:
                    #     logger.info(
                    #         "frame #{},time:{}μs,get_img_number:{}".format(
                    #             frame.frame_count,
                    #             round(frame.time_stamp_relative_ns_or_null * (10 ** -3)),
                    #             i_get_img_number
                    #         )
                    #     )
                else:
                    logger.error("timeout reached during polling, program exiting...")
                    break

            DMD.Wait()
            camera.disarm()

            DMD.Halt()
            DMD.FreeSeq()

            # get 2d TM array
            TM = camimg.createTM_cam(camera_img, camera_length_afterDownSample)
            # get a group of 3d phase arrays and a group of corresponding 3d amplitude arrays from 2d array TM
            TM_phase_array, TM_amplitude_array = camimg.TM_2_phase_amplitde_array(TM)
            logger.info("amplitude max:{} min:{}".format(np.max(TM_amplitude_array), np.min(TM_amplitude_array)))

            file_name_tm = f'MMF_{hadamardSize}hadamard_{camera_length_afterDownSample}camera_img_size.npy'
            np.save('./measure_tm/' + file_name_tm, TM)
            logger.info("save TM to {}".format(file_name_tm))

            # 裁剪数组，只扫描200个点
            block_size_loop_focus_put = 200
            TM_phase_array = TM_phase_array[2000:2200]
            TM_amplitude_array = TM_amplitude_array[2000:2200]


            # project by DMD
            focusnumber = TM_phase_array.shape[0]
            DMD.SeqAlloc(nbImg=focusnumber, bitDepth=bitDepth)
            DMD.SeqControl(ALP_BITNUM, 1)  # <bit number> 1 … BitPlanes
            DMD.SeqControl(
                ALP_BIN_MODE, ALP_BIN_UNINTERRUPTED
            )  # Operation without dark phase_4
            DMD.SeqControl(
                ALP_SEQ_REPEAT, repeat_focus
            )  # the sequence is displayed this number of times

            DMD.SetTiming(
                illuminationTime=illuminationTime_focus,
                pictureTime=pictureTime_focus,
                synchPulseWidth=100,
            )  # μs

            # Generate patterns
            for m_put in range(0, TM_phase_array.shape[0], block_size_loop_focus_put):
                logger.info("------------loading put focus number {}-----------".format(m_put))
                start_focus_put = m_put
                end_focus_put = min(m_put + block_size_loop_focus_put, TM_phase_array.shape[0])
                current_TM_phase_array = TM_phase_array[start_focus_put:end_focus_put]
                current_TM_amplitude_array = TM_amplitude_array[start_focus_put:end_focus_put]
                if axial_method == 1:
                    current_TM_focus_array = dmdpat.getFocusProject_phase_amplitude_coaxial(
                        superpixel, current_TM_phase_array, current_TM_amplitude_array,
                        holo_method, alpha=alpha, x0=x0, rot=rot)
                elif axial_method == 2:
                    current_TM_focus_array = dmdpat.getFocusProject_phase_amplitude_offaxial(
                        superpixel, current_TM_phase_array, current_TM_amplitude_array,
                        holo_method, alpha=alpha, x0=x0, rot=rot)
                else:
                    logger.error("Must choose axial_methold 1: coaxial or 2: offaxial")
                    raise ValueError("Must choose axial_methold 1: coaxial or 2: offaxial")
                DMD.SeqPut(imgData=current_TM_focus_array, PicOffset= m_put,
                           PicLoad=current_TM_focus_array.shape[0])

            camera.arm(2)
            camera_img_focus = np.zeros((block_size_loop_focus_put, camera_length, camera_length), dtype=np.uint16)
            i_get_img_number_focus = 0

            logger.info("Start focus projection")
            DMD.Run(loop=False)

            for i_camfocus in range(block_size_loop_focus_put):
                frame_focus = camera.get_pending_frame_or_null()
                if frame_focus is not None:
                    # print("frame rate:{}".format(camera.get_measured_frame_rate_fps()))
                    # camera_img[i_cam] = np.copy(frame.image_buffer)
                    # 相机采集图像数值是10bits，所以是0-1023的整数
                    camera_img_focus[i_camfocus] = frame_focus.image_buffer[cam_leftuppery:cam_leftuppery + camera_length,
                                        cam_leftupperx:cam_leftupperx + camera_length]
                    i_get_img_number_focus += 1
                    if i_get_img_number_focus % 10 == 0:
                        logger.info(
                            "frame_focus #{},time:{}μs,get_img_number:{}".format(
                                frame_focus.frame_count,
                                round(frame_focus.time_stamp_relative_ns_or_null * (10 ** -3)),
                                i_get_img_number
                            )
                        )
                else:
                    logger.error("timeout reached during polling, program exiting...")
                    break

            DMD.Wait()
            camera.disarm()

# plot image
for i_img in range(4):
    plt.figure(i_img)
    plt.imshow(
        camera_img_focus[i_img+100],
        cmap="binary",
        origin="lower",
    )
    plt.colorbar(label="binary")
    plt.title("image to project{}".format(i_img+100))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)
plt.show()

