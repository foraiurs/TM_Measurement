#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/5/27 16:24
@author: Zhen Cheng
"""

"""
dmd投图后，用来看看散斑的样子以及resize后的结果
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from DMD_holography import *

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from cmos_examples import configure_path

    configure_path()
except ImportError:
    configure_path = None

import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

NUM_FRAMES = 3  # adjust to the desired number of frames


with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    print("serial number:{}".format(available_cameras))
    if len(available_cameras) < 1:
        print("no cameras detected")

    with sdk.open_camera(available_cameras[0]) as camera:
        camera.exposure_time_us = 100  # set exposure μs
        camera.frames_per_trigger_zero_for_unlimited = (
            0  # 0:continuous mode or the number of frames after trigger
        )
        camera.image_poll_timeout_ms = 1000  # polling timeout ms 等待时间，过后没有采到图就停止
        # camera.roi = (0, 0, 1279, 1023)  # store the current roi
        camera.roi = (560, 370, 760, 570)
        camera_length = 200
        tm_img = 64
        camimg = ImgAcquire()
        cam_leftuppery, cam_leftupperx = camimg.getparameter_cam(camera.roi, camera_length)
        camera.operation_mode = 0
        print(camera.roi[0], camera.roi[1], camera.roi[2], camera.roi[3])
        print(camera.roi_range)
        camera_img = np.zeros((NUM_FRAMES, camera_length, camera_length), dtype=np.uint16)
        """
        uncomment the line below to set a region of interest (ROI) on the camera
        """
        # camera.roi = (100, 100, 600, 600)  # set roi to be at origin point (100, 100) with a width & height of 500

        """
        uncomment the lines below to set the gain of the camera and read it back in decibels
        """
        # if camera.gain_range.max > 0:
        #    db_gain = 6.0
        #    gain_index = camera.convert_decibels_to_gain(db_gain)
        #    camera.gain = gain_index
        #    print(f"Set camera gain to {camera.convert_gain_to_decibels(camera.gain)}")
        """
        camera采集图像的过程：
        arm(frames_to_buffer)准备采集，frames_to_buffer表示在software/hardware trigger后把几个frames放入
        buffer，最小值为2，通过调用operation_mode可以选择software(=0)还是hardware(=1),BULB(=2) trigger，下面分类讨论
        对software trigger : 调用operation_mode选择software(=0), 若frames_per_trigger_zero_for_unlimited = 0，即连续模式，此时调用
        issue_software_trigger，相当于一次software trigger后，相机将无限地连续采集frames，之间的间隔为exposure_time；
        若frames_per_trigger_zero_for_unlimited = m，则一次software trigger后，相机将采集m个frames，之间间隔为
        exposure_time+10ms。对于形成的frames列，camera.issue_software_trigger()后，根据arm时设置的frames_to_buffer=i，
        相机会将前i个frames放入buffer，当buffer中的这i个frames被抓取后，相机会将此刻的frames送入buffer替换这i个frames，
        然后在exposure_time后用新的下一个frame替换 buffer里的frame，不断循环，调用get_pending_frame_or_null，
        它会抓取buffer里的frames，调用一次返回抓取的一个
        frame(这个frame类Frame的一个实例), 在image_poll_timeout_ms时间内如果buffer里没有可抓取的frame，则返回None
        对于类Frame，有几个属性，image_buffer是表示图像的numpy数组np.array，frame_count是frame的序号，time_stamp_relative_ns_or_null
        表示每一帧生成的时间(ns)
        对hardware/bulb trigger, 调用operation_mode选择hardware(=1) or bulb(=2), frames_per_trigger_zero_for_unlimited设置=1，
        trigger_polarity设置rising-edge(=0) or falling-edge(=1) triggered  (bulb:使用硬件触发并使用该信号确定曝光时间)，这时不论arm设置的
        多少，都是等待外部触发，外部触发后采集进
        相机采集图像数值是10bits，所以是0-1023的整数
        """

        print("data rate:{}".format(camera.get_is_data_rate_supported(2)))
        print(
            "if hardware trigger support:{}".format(
                camera.get_is_operation_mode_supported(1)
            )
        )
        print("color matrix:{}".format(camera.get_color_correction_matrix()))
        print("clock time:{}".format(camera._get_time_stamp_clock_frequency_or_null()))
        print(
            "operation mode:{} (SOFTWARE_TRIGGERED = 0,HARDWARE_TRIGGERED = 1)".format(
                camera.operation_mode
            )
        )
        print("trigger range:{}".format(camera.frames_per_trigger_range))
        print("explore time range(μs):{}".format(camera.exposure_time_range_us))
        print("interval between two triggers(μs):{}".format(camera.frame_time_us))
        print("binx size:{}".format(camera.binx))
        print(
            "time(ns) that readout data from sensor:{}".format(
                camera.sensor_readout_time_ns
            )
        )
        print("binx range:{}".format(camera.binx_range))
        print(
            camera.model, camera.name, camera.is_eep_supported, camera.is_led_supported
        )
        print(camera.camera_sensor_type)
        print("bit depth:{}".format(camera.bit_depth))
        print("camera.gain_range:{}".format(camera.gain_range))
        camera.gain = 2
        print("gain:{}".format(camera.gain))
        for g in range(0, 4):
            print("gain(dB)={}:{}dB".format(g, camera.convert_gain_to_decibels(g)))

        camera.arm(2)
        camera.issue_software_trigger()
        for i in range(NUM_FRAMES):
            # time.sleep(10)
            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                print(
                    "---------------frame #{} received!--------------------".format(
                        frame.frame_count
                    )
                )
                print("frame rate:{}".format(camera.get_measured_frame_rate_fps()))
                print(
                    "frame time:{}μs".format(
                        round((frame.time_stamp_relative_ns_or_null) * (10**-3))
                    )
                )
                print(frame.image_buffer.shape)
                print(
                    frame.image_buffer
                )  # .../ perform operations using the data from image_buffer
                camera_img[i] = frame.image_buffer[cam_leftuppery:cam_leftuppery + camera_length,
                                    cam_leftupperx:cam_leftupperx + camera_length]
            else:
                print("timeout reached during polling, program exiting...")
                break

        camera.disarm()

        camera.arm(2)
        camera.issue_software_trigger()
        for i in range(NUM_FRAMES):
            # time.sleep(10)
            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                print(
                    "---------------frame #{} received!--------------------".format(
                        frame.frame_count
                    )
                )
                print("frame rate:{}".format(camera.get_measured_frame_rate_fps()))
                print(
                    "frame time:{}μs".format(
                        round((frame.time_stamp_relative_ns_or_null) * (10 ** -3))
                    )
                )
                print(frame.image_buffer.shape)
                print(
                    frame.image_buffer
                )  # .../ perform operations using the data from image_buffer
                camera_img[i] = frame.image_buffer[cam_leftuppery:cam_leftuppery + camera_length,
                                cam_leftupperx:cam_leftupperx + camera_length]
            else:
                print("timeout reached during polling, program exiting...")
                break
        camera.disarm()



for i_img in range(NUM_FRAMES):
    plt.figure(i_img)
    plt.imshow(
        camera_img[i_img],
        cmap="binary",
        origin="lower",
    )
    plt.colorbar(label="frame.image_buffer")
    plt.title("image{}".format(i_img))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)


img_z = camera_img.shape[0]
camera_img_re = np.zeros((img_z, tm_img, tm_img))

print("before resize", camera_img, camera_img.shape)
for i_re in range(img_z):
    camera_img_re[i_re] = resize(camera_img[i_re], (tm_img, tm_img), order=1, preserve_range=True, anti_aliasing=True)
print("after resize", camera_img_re, camera_img_re.shape)


for i_img in range(NUM_FRAMES):
    plt.figure(i_img+NUM_FRAMES)
    plt.imshow(
        camera_img_re[i_img],
        cmap="binary",
        origin="lower",
    )
    plt.colorbar(label="frame.image_buffer_resize")
    plt.title("image{}".format(i_img+NUM_FRAMES))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(0)
plt.show()


#  Because we are using the 'with' statement context-manager, disposal has been taken care of.

print("program completed")
