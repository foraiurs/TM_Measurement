#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/7/13 14:13
@author: Zhen Cheng
"""

import numpy as np
from Patterncreate import *
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.transform import resize
from typing import Union, Iterable
from sklearn.preprocessing import MinMaxScaler
import numba as nb
import code_tools


logger = code_tools.get_logger(__name__)


class PatternProject(object):
    """
    Generate DMD patterns
    """

    def __init__(self, DMDwidth=1920, DMDheight=1080):
        self._DMDwidth = DMDwidth
        self._DMDheight = DMDheight


    def getparameter(self, array_length=16, referenceFraction=0.35):
        """
        Parameters required for generating patterns

        :param referenceFraction: The approximate proportion of the reference area, not the actual value
        :return: each hadamard "pixel" is consisted of BlockSize*BlockSize pixels
        effectiveDMDsize = phase_array_length*BlockSize+2*numReferencePixels

        """
        effectiveDMDsize = min(self._DMDheight, self._DMDwidth)
        numReferencePixels = round(
            (effectiveDMDsize - np.sqrt(1 - referenceFraction) * effectiveDMDsize) / 2
        )
        BlockSize = int(np.floor(((effectiveDMDsize - 2 * numReferencePixels) / array_length)))
        numReferencePixels = (effectiveDMDsize - BlockSize * array_length) / 2
        signalRatio = ((BlockSize * array_length) ** 2) / (
                effectiveDMDsize * effectiveDMDsize
        )
        logger.info(
            "{}*{}DMD. Using {}*{} basis. Each uses {}x{} pixels. Reference is {} pixels. Reference area accounts for "
            "{}".format(
                self._DMDwidth,
                self._DMDheight,
                array_length,
                array_length,
                BlockSize,
                BlockSize,
                numReferencePixels,
                1 - signalRatio,
            )
        )
        return BlockSize

    def get_hadamardBasis_phaseArrray(self, hadamardSize=16):
        """
        Generate a 3D Hadamard phase_4 pattern
        """

        pat_had = Pattern()
        hadphasePatterns = pat_had.HadamardPattern(
            hadamardSize
        )
        hadphasePatterns = np.where(
            hadphasePatterns == 1, 0, np.pi
        )  # replace 1 with 0, -1 with π
        return hadphasePatterns

    # @nb.jit(nopython=True, parallel=True)
    def getGratingProject_phase_coaxial(
            self,
            BlockSize: Union[int, Iterable],
            phase_array: Union[np.ndarray, Iterable],
            phase_4=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
            methold=1,
            **kwargs
    ):
        """
        Supports any step phase shift

        gt a 3D phase_4 array ->
        Due to memory size limitations, use a for loop->
        Take Kronecker product between each 2D array and a leeBlockSize*leeBlockSize matrix consisting of all ones ->
        Repeat each 2D pattern four times ->
        Add respective phase_4 shifts corresponding to the four steps(0, π/2, π, 3*π/2)->
        Add a reference phase_4 of 0 to each pattern to form a square ->
        Generate hologram(boyd=1/lee=2)

        :return: 3D (np.size(phase_4)*n**2)*DMDwidth*DMDheight holograph_patterns

        """

        # Set some parameters
        pat = Pattern()
        phase_array_z, phase_array_length, _ = phase_array.shape
        effectiveDMDsize = min(self._DMDheight, self._DMDwidth)
        phasenumber = np.size(phase_4)
        DMDextraleft = round((self._DMDwidth - effectiveDMDsize) / 2)
        DMDextraright = self._DMDwidth - effectiveDMDsize - DMDextraleft
        numReferencePixelsleft = round(
            (effectiveDMDsize - BlockSize * phase_array_length) / 2
        )
        numReferencePixelsright = (
                effectiveDMDsize - BlockSize * phase_array_length - numReferencePixelsleft
        )

        # Due to memory size limitations, use a for loop
        block_size_loop = 50  # the number of z every loop
        result_array = np.zeros(
            (phasenumber * phase_array_z, self._DMDheight, self._DMDwidth), dtype=np.uint8
        )

        for i_holo in range(0, phase_array_z, block_size_loop):
            logger.info("--------loading phase_array_z={}----------".format(i_holo))
            start_holo = i_holo
            end_holo = min(i_holo + block_size_loop, phase_array_z)
            currentPatterns = phase_array[start_holo:end_holo]

            block_size_norepeat = currentPatterns.shape[0]
            # Take Kronecker product
            currentPatterns = np.kron(
                currentPatterns, np.ones((BlockSize, BlockSize))
            )
            currentPatterns = np.repeat(
                currentPatterns, phasenumber, axis=0
            )  # Repeat each 2D pattern four times
            # logger.info(
            #     "binaryBasis4_kron shape:{}, dtype:{}".format(
            #         currentPatterns.shape, currentPatterns.dtype
            #     )
            # )

            # LeeHoloPatterns = np.array(LeeHoloPatterns, dtype=np.float16)
            block_size_repeat = currentPatterns.shape[0]
            holo_angles = np.tile(phase_4, block_size_norepeat)
            holo_angles_broadcast = holo_angles.reshape((block_size_repeat, 1, 1))
            # holo_angles_broadcast = np.array(holo_angles_broadcast, dtype=np.float16)
            # logger.info(
            #     "holo_angles_broadcast shape:{}, dtype:{}".format(
            #         holo_angles_broadcast.shape, currentPatterns.dtype
            #     )
            # )
            currentPatterns += holo_angles_broadcast  # Add respective phase_4 shifts corresponding to the four steps(0, π/2, π, 3*π/2)
            #  LeeHoloPatterns shape: (4*n**2)*(n*leeBlockSize)*(n*leeBlockSize)

            # Add a reference phase_4 of π/2
            currentPatterns = np.pad(
                currentPatterns,
                (
                    (0, 0),
                    (numReferencePixelsleft, numReferencePixelsright),
                    (
                        numReferencePixelsleft + DMDextraleft,
                        numReferencePixelsright + DMDextraright,
                    ),
                ),
                "constant",
                constant_values=0,
            )
            # logger.info(
            #     "phaseBasis4ref shape:{}, dtype:{}".format(
            #         currentPatterns.shape, currentPatterns.dtype
            #     )
            # )
            # LeeHoloPatterns shape: (4*n**2)*(effectiveDMDsize)*(effectiveDMDsize)

            # Generate Lee hologram
            if methold == 1:
                amplitude = kwargs.get("amplitude", 1)
                alpha = kwargs.get("alpha", 0.1)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.LeePattern(currentPatterns, amplitude, alpha, rot)

            # Generate boyd hologram
            elif methold == 2:
                amplitude = kwargs.get("amplitude", 1)
                x0 = kwargs.get("x0", 10)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.BoydPattern(currentPatterns, amplitude, x0, rot)

            else:
                logger.error("Must choose methold, 1:Lee, 2:Boyd")
                raise ValueError("Must choose methold, 1:Lee, 2:Boyd")

            result_array[phasenumber * start_holo: phasenumber * end_holo] = (
                    currentPatterns * 255
            )

        logger.info(
            "Leehol_binarydmd shape:{}, dtype:{}".format(
                result_array.shape, result_array.dtype
            )
        )
        return result_array

    # @nb.jit(nopython=True, parallel=True)
    def getGratingProject_phase_offaxial(
            self,
            BlockSize: Union[int, Iterable],
            phase_array: Union[np.ndarray, Iterable],
            phase_4=(0, np.pi / 2, np.pi, 3 * np.pi / 2),
            methold=1,
            **kwargs
    ):
        """
        Supports any step phase_4 shift

        gt a 3D phase_4 array ->
        Due to memory size limitations, use a for loop->
        Take Kronecker product between each 2D array and a leeBlockSize*leeBlockSize matrix consisting of all ones ->
        Repeat each 2D pattern four times ->
        Add respective phase_4 shifts corresponding to the four steps(0, π/2, π, 3*π/2)->
        Generate hologram(boyd=1/lee=2)
        Add a reference phase_4 of 0 to each pattern to form a square

        :return: 3D (np.size(phase_4)*n**2)*DMDwidth*DMDheight holograph_patterns

        """

        # Set some parameters
        pat = Pattern()
        phase_array_z, phase_array_length, _ = phase_array.shape
        effectiveDMDsize = min(self._DMDheight, self._DMDwidth)
        phasenumber = np.size(phase_4)
        DMDextraleft = round((self._DMDwidth - effectiveDMDsize) / 2)
        DMDextraright = self._DMDwidth - effectiveDMDsize - DMDextraleft
        numReferencePixelsleft = round(
            (effectiveDMDsize - BlockSize * phase_array_length) / 2
        )
        numReferencePixelsright = (
                effectiveDMDsize - BlockSize * phase_array_length - numReferencePixelsleft
        )

        # Due to memory size limitations, use a for loop
        block_size_loop = 50  # the number of z every loop
        result_array = np.zeros(
            (phasenumber * phase_array_z, self._DMDheight, self._DMDwidth), dtype=np.uint8
        )

        for i_holo in range(0, phase_array_z, block_size_loop):
            logger.info("--------loading phase_array_z={}----------".format(i_holo))
            start_holo = i_holo
            end_holo = min(i_holo + block_size_loop, phase_array_z)
            currentPatterns = phase_array[start_holo:end_holo]

            block_size_norepeat = currentPatterns.shape[0]
            # Take Kronecker product
            currentPatterns = np.kron(
                currentPatterns, np.ones((BlockSize, BlockSize))
            )
            currentPatterns = np.repeat(
                currentPatterns, phasenumber, axis=0
            )  # Repeat each 2D pattern four times
            # logger.info(
            #     "binaryBasis4_kron shape:{}, dtype:{}".format(
            #         currentPatterns.shape, currentPatterns.dtype
            #     )
            # )

            # LeeHoloPatterns = np.array(LeeHoloPatterns, dtype=np.float16)
            block_size_repeat = currentPatterns.shape[0]
            holo_angles = np.tile(phase_4, block_size_norepeat)
            holo_angles_broadcast = holo_angles.reshape((block_size_repeat, 1, 1))
            # holo_angles_broadcast = np.array(holo_angles_broadcast, dtype=np.float16)
            # logger.info(
            #     "holo_angles_broadcast shape:{}, dtype:{}".format(
            #         holo_angles_broadcast.shape, currentPatterns.dtype
            #     )
            # )
            currentPatterns += holo_angles_broadcast  # Add respective phase_4 shifts corresponding to the four steps(0, π/2, π, 3*π/2)
            #  LeeHoloPatterns shape: (4*n**2)*(n*leeBlockSize)*(n*leeBlockSize)

            # Generate Lee hologram
            if methold == 1:
                amplitude = kwargs.get("amplitude", 1)
                alpha = kwargs.get("alpha", 0.1)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.LeePattern(currentPatterns, amplitude, alpha, rot)

            # Generate boyd hologram
            elif methold == 2:
                amplitude = kwargs.get("amplitude", 1)
                x0 = kwargs.get("x0", 10)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.BoydPattern(currentPatterns, amplitude, x0, rot)

            else:
                logger.error("Must choose methold, 1:Lee, 2:Boyd")
                raise ValueError("Must choose methold, 1:Lee, 2:Boyd")

            # Add 0 to the size of DMD
            currentPatterns = np.pad(
                currentPatterns,
                (
                    (0, 0),
                    (numReferencePixelsleft, numReferencePixelsright),
                    (
                        numReferencePixelsleft + DMDextraleft,
                        numReferencePixelsright + DMDextraright,
                    ),
                ),
                "constant",
                constant_values=0,
            )
            # logger.info(
            #     "phaseBasis4ref shape:{}, dtype:{}".format(
            #         currentPatterns.shape, currentPatterns.dtype
            #     )
            # )
            # LeeHoloPatterns shape: (4*n**2)*(effectiveDMDsize)*(effectiveDMDsize)

            result_array[phasenumber * start_holo: phasenumber * end_holo] = (
                    currentPatterns * 255
            )

        logger.info(
            "Leehol_binarydmd shape:{}, dtype:{}".format(
                result_array.shape, result_array.dtype
            )
        )
        return result_array

    # @nb.jit(nopython=True, parallel=True)
    def getFocusProject_phase_amplitude_coaxial(
            self,
            BlockSize: Union[int, Iterable],
            phase_array: Union[np.ndarray, Iterable],
            amplitude_array: Union[np.ndarray, Iterable],
            methold=1,
            **kwargs
    ):
        """

        amplitude_array should be 0-1

        gt a 3D focus array ->
        Due to memory size limitations, use a for loop->
        Take Kronecker product between each 2D array and a leeBlockSize*leeBlockSize matrix consisting of all ones ->
        Add a reference 0 for phase and 1 for amplitude ->
        Generate hologram(boyd=1/lee=2)

        :return: 3D (n**2)*DMDwidth*DMDheight holograph_patterns

        """

        # Set some parameters
        pat = Pattern()
        phase_array_z, phase_array_length, _ = phase_array.shape
        effectiveDMDsize = min(self._DMDheight, self._DMDwidth)
        DMDextraleft = round((self._DMDwidth - effectiveDMDsize) / 2)
        DMDextraright = self._DMDwidth - effectiveDMDsize - DMDextraleft
        numReferencePixelsleft = round(
            (effectiveDMDsize - BlockSize * phase_array_length) / 2
        )
        numReferencePixelsright = (
                effectiveDMDsize - BlockSize * phase_array_length - numReferencePixelsleft
        )

        # Due to memory size limitations, use a for loop
        block_size_loop = 200  # the number of z every loop
        result_array = np.zeros(
            (phase_array_z, self._DMDheight, self._DMDwidth), dtype=np.uint8
        )

        for i_holo in range(0, phase_array_z, block_size_loop):
            logger.info("--------loading focus_array_z={}----------".format(i_holo))
            start_holo = i_holo
            end_holo = min(i_holo + block_size_loop, phase_array_z)
            currentPatterns_phase = phase_array[start_holo:end_holo]
            currentPatterns_amplitude = amplitude_array[start_holo:end_holo]

            block_size_real = currentPatterns_phase.shape[0]
            # Take Kronecker product
            currentPatterns_phase = np.kron(
                currentPatterns_phase, np.ones((BlockSize, BlockSize))
            )
            currentPatterns_amplitude = np.kron(
                currentPatterns_amplitude, np.ones((BlockSize, BlockSize))
            )
            # logger.info(
            #     "binaryBasis4_kron shape phase_4:{}, dtype:{}".format(
            #         currentPatterns_phase.shape, currentPatterns_phase.dtype
            #     )
            # )
            # logger.info(
            #     "binaryBasis4_kron shape amplitude:{}, dtype:{}".format(
            #         currentPatterns_amplitude.shape, currentPatterns_amplitude.dtype
            #     )
            # )

            #  LeeHoloPatterns shape: (n**2)*(n*leeBlockSize)*(n*leeBlockSize)

            # Add a reference 0 for phase and 1 for amplitude
            currentPatterns_phase = np.pad(
                currentPatterns_phase,
                (
                    (0, 0),
                    (numReferencePixelsleft, numReferencePixelsright),
                    (
                        numReferencePixelsleft + DMDextraleft,
                        numReferencePixelsright + DMDextraright,
                    ),
                ),
                "constant",
                constant_values=0,
            )
            # logger.info(
            #     "phaseBasis4ref shape phase:{}, dtype:{}".format(
            #         currentPatterns_phase.shape, currentPatterns_phase.dtype
            #     )
            # )

            currentPatterns_amplitude = np.pad(
                currentPatterns_amplitude,
                (
                    (0, 0),
                    (numReferencePixelsleft, numReferencePixelsright),
                    (
                        numReferencePixelsleft + DMDextraleft,
                        numReferencePixelsright + DMDextraright,
                    ),
                ),
                "constant",
                constant_values=1,
            )
            # logger.info(
            #     "phaseBasis4ref shape amplitude:{}, dtype:{}".format(
            #         currentPatterns_amplitude.shape, currentPatterns_amplitude.dtype
            #     )
            # )
            # LeeHoloPatterns shape: (n**2)*(effectiveDMDsize)*(effectiveDMDsize)

            # Generate Lee hologram
            if methold == 1:
                alpha = kwargs.get("alpha", 0.1)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.LeePattern(currentPatterns_phase, currentPatterns_amplitude, alpha, rot)

            # Generate boyd hologram
            elif methold == 2:
                x0 = kwargs.get("x0", 10)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.BoydPattern(currentPatterns_phase, currentPatterns_amplitude, x0, rot)

            else:
                logger.error("Must choose methold, 1:Lee, 2:Boyd")
                raise ValueError("Must choose methold, 1:Lee, 2:Boyd")

            result_array[start_holo: end_holo] = (
                    currentPatterns * 255
            )

        logger.info(
            "Leehol_binarydmd shape:{}, dtype:{}".format(
                result_array.shape, result_array.dtype
            )
        )
        return result_array

    # @nb.jit(nopython=True, parallel=True)
    def getFocusProject_phase_amplitude_offaxial(
            self,
            BlockSize: Union[int, Iterable],
            phase_array: Union[np.ndarray, Iterable],
            amplitude_array: Union[np.ndarray, Iterable],
            methold=1,
            **kwargs
    ):
        """
        amplitude_array should be 0-1

        gt a 3D array ->
        Due to memory size limitations, use a for loop->
        Take Kronecker product between each 2D array and a leeBlockSize*leeBlockSize matrix consisting of all ones ->
        Generate hologram(boyd=1/lee=2)
        Add a reference of 0 to each pattern to form a square

        :return: 3D (n**2)*DMDwidth*DMDheight holograph_patterns

        """

        # Set some parameters
        pat = Pattern()
        phase_array_z, phase_array_length, _ = phase_array.shape
        effectiveDMDsize = min(self._DMDheight, self._DMDwidth)
        DMDextraleft = round((self._DMDwidth - effectiveDMDsize) / 2)
        DMDextraright = self._DMDwidth - effectiveDMDsize - DMDextraleft
        numReferencePixelsleft = round(
            (effectiveDMDsize - BlockSize * phase_array_length) / 2
        )
        numReferencePixelsright = (
                effectiveDMDsize - BlockSize * phase_array_length - numReferencePixelsleft
        )

        # Due to memory size limitations, use a for loop
        block_size_loop = 200  # the number of z every loop
        result_array = np.zeros(
            (phase_array_z, self._DMDheight, self._DMDwidth), dtype=np.uint8
        )

        for i_holo in range(0, phase_array_z, block_size_loop):
            logger.info("--------loading phase_array_z={}----------".format(i_holo))
            start_holo = i_holo
            end_holo = min(i_holo + block_size_loop, phase_array_z)
            currentPatterns_phase = phase_array[start_holo:end_holo]
            currentPatterns_amplitude = amplitude_array[start_holo:end_holo]

            block_size_real = currentPatterns_phase.shape[0]
            # Take Kronecker product
            currentPatterns_phase = np.kron(
                currentPatterns_phase, np.ones((BlockSize, BlockSize))
            )
            currentPatterns_amplitude = np.kron(
                currentPatterns_amplitude, np.ones((BlockSize, BlockSize))
            )
            # logger.info(
            #     "binaryBasis4_kron shape phase_4:{}, dtype:{}".format(
            #         currentPatterns_phase.shape, currentPatterns_phase.dtype
            #     )
            # )
            # logger.info(
            #     "binaryBasis4_kron shape amplitude:{}, dtype:{}".format(
            #         currentPatterns_amplitude.shape, currentPatterns_amplitude.dtype
            #     )
            # )

            #  LeeHoloPatterns shape: (n**2)*(n*leeBlockSize)*(n*leeBlockSize)

            # Generate Lee hologram
            if methold == 1:
                alpha = kwargs.get("alpha", 0.1)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.LeePattern(currentPatterns_phase, currentPatterns_amplitude, alpha, rot)

            # Generate boyd hologram
            elif methold == 2:
                x0 = kwargs.get("x0", 10)
                rot = kwargs.get("rot", 7 * np.pi / 4)
                currentPatterns = pat.BoydPattern(currentPatterns_phase, currentPatterns_amplitude, x0, rot)

            else:
                logger.error("Must choose methold, 1:Lee, 2:Boyd")
                raise ValueError("Must choose methold, 1:Lee, 2:Boyd")

            # Add 0 to the size of DMD
            currentPatterns = np.pad(
                currentPatterns,
                (
                    (0, 0),
                    (numReferencePixelsleft, numReferencePixelsright),
                    (
                        numReferencePixelsleft + DMDextraleft,
                        numReferencePixelsright + DMDextraright,
                    ),
                ),
                "constant",
                constant_values=0,
            )
            # logger.info(
            #     "phaseBasis4ref shape:{}, dtype:{}".format(
            #         currentPatterns.shape, currentPatterns.dtype
            #     )
            # )
            # LeeHoloPatterns shape: (4*n**2)*(effectiveDMDsize)*(effectiveDMDsize)

            result_array[start_holo: end_holo] = (
                    currentPatterns * 255
            )

        logger.info(
            "Leehol_binarydmd shape:{}, dtype:{}".format(
                result_array.shape, result_array.dtype
            )
        )
        return result_array


class ImgAcquire(object):
    """
    Camera captures images and processes
    """

    def __init__(self):
        pass

    def getparameter_cam(self, camera_roi, camera_length=512):
        """
        The cropped parameter of camera imagine
        """
        img_height = camera_roi[3] - camera_roi[1] + 1
        img_width = camera_roi[2] - camera_roi[0] + 1
        left_upper_y = round((img_height - camera_length) / 2)
        left_upper_x = round((img_width - camera_length) / 2)
        logger.info(
            "cam_roi_nocropped:{}, cam_left_upper_y:{}, cam_left_upper_x:{}".format(camera_roi,
                                                                                    left_upper_y, left_upper_x)
        )
        return left_upper_y, left_upper_x

    # @nb.jit(nopython=True, parallel=True)
    def createTM_cam(self, camera_img, tm_img=64):
        """
        four-step phase_4-shifting
        """

        img_z, img_length, _ = camera_img.shape
        tm_x = img_z // 4
        tm_array = np.zeros(
            (tm_img ** 2, tm_x), dtype=np.complex64)
        block_size_loop_tm = 300
        block_size_loop_tm4 = 4 * block_size_loop_tm

        # 注意数组z轴长度最后一个循环的时候不是block_size_loop_tm4
        for i_tm in range(0, img_z, block_size_loop_tm4):
            logger.info("--------loading img_z={}----------".format(i_tm))
            start_tm = i_tm
            end_tm = min(i_tm + block_size_loop_tm4, img_z)
            currentImg = camera_img[start_tm:end_tm]
            currentImg_z = currentImg.shape[0]

            # resize
            if tm_img != img_length:
                camera_img_re = np.zeros((currentImg_z, tm_img, tm_img))
                for i_re in range(currentImg_z):
                    camera_img_re[i_re] = resize(currentImg[i_re], (tm_img, tm_img),
                                                 order=0, preserve_range=True, anti_aliasing=True)
                currentImg = camera_img_re

            # 数组（block_size_loop_tm4， tm_img， tm_img）
            reshaped_currentImg = currentImg.reshape((currentImg_z, tm_img ** 2))
            indexes_current = np.arange(0, currentImg_z, 4)
            real_part_tm = (reshaped_currentImg[indexes_current] - reshaped_currentImg[indexes_current + 2]) / 4
            imag_part_tm = (reshaped_currentImg[indexes_current + 3] - reshaped_currentImg[indexes_current + 1]) / 4
            tm_array[:, start_tm // 4: end_tm // 4] = (real_part_tm + 1j * imag_part_tm).T

        tm_array = np.dot(tm_array, hadamard(tm_x))
        logger.info(f"tm_array{tm_array.shape}{tm_array.dtype}")
        return tm_array

    # @nb.jit(nopython=True, parallel=True)
    def TM_2_phase_amplitde_array(self, TM: Union[np.ndarray, Iterable]):
        """
        Get a group of 3d phase arrays and a group of corresponding 3d amplitude arrays from 2d array TM
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        TM_y, TM_x = TM.shape
        array_length = int(np.sqrt(TM_x))
        TM = np.conj(TM)
        TM_phase_array = np.angle(TM)
        TM_amplitude_array = np.abs(TM)
        TM_phase_array = TM_phase_array.reshape((TM_y, array_length, array_length))  #.astype(np.float32)
        TM_amplitude_array = scaler.fit_transform(TM_amplitude_array.T).T.reshape(
            (TM_y, array_length, array_length))  #.astype(np.float32)
        logger.info(f"TM_phase_array:{TM_phase_array.shape}, TM_amplitude_array:{TM_amplitude_array.dtype}")
        # 利用了MinMaxScaler对于二维数组是对其每一列做缩放的特性，实现对于每一幅hadamardsize*hadamardsize的图单独做0到1的缩放

        return TM_phase_array, TM_amplitude_array


if __name__ == "__main__":
    hadamardSize = 16
    phasetest = (0, np.pi / 2, np.pi, 3 * np.pi / 2)

    # pattern create
    holotest = PatternProject(1920, 1080)
    BlockSize = holotest.getparameter(hadamardSize, 0.35)
    phase_array = holotest.get_hadamardBasis_phaseArrray(hadamardSize)
    patterns = holotest.getGratingProject_phase_offaxial(BlockSize, phase_array, phasetest, 1, amplitude=1, alpha=0.1,
                                                         x0=10, rot=7 * np.pi / 4)
    print("patterns shape:{},dtype:{}".format(patterns.shape, patterns.dtype))
    # for row in patterns[100, :, :]:
    #     for element in row:
    #         print(element, end=' ')
    #     print()  # 打印完一行后换行

    hadpat = Pattern()
    binaryBasis = hadpat.HadamardPattern(hadamardSize)  # Generate a 3D Hadamard pattern
    effectiveDMDsize = min(1080, 1920)
    DMDextraleft = round((1920 - effectiveDMDsize) / 2)
    DMDextraright = 1920 - effectiveDMDsize - DMDextraleft
    numReferencePixelsleft = round((effectiveDMDsize - BlockSize * hadamardSize) / 2)
    numReferencePixelsright = (
            effectiveDMDsize - BlockSize * hadamardSize - numReferencePixelsleft
    )
    binaryBasis4 = np.repeat(
        binaryBasis, np.size(phasetest), axis=0
    )  # Repeat each 2D pattern four times

    for i_had in range(0, 4):
        plt.figure(i_had + 1)
        binaryBasis4_1 = binaryBasis4[i_had + 1019, :, :]
        binaryBasis4_1 = np.kron(
            binaryBasis4_1, np.ones((BlockSize, BlockSize), dtype=np.int8)
        )
        binaryBasis4_1 = np.pad(
            binaryBasis4_1,
            (
                (numReferencePixelsleft, numReferencePixelsright),
                (
                    numReferencePixelsleft + DMDextraleft,
                    numReferencePixelsright + DMDextraright,
                ),
            ),
            "constant",
            constant_values=0,
        )

        plt.imshow(binaryBasis4_1, cmap="viridis", origin="lower")
        plt.colorbar()
        plt.title("hadpat{}".format(i_had + 1019))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(0)

    for i_lee in range(0, 4):
        plt.figure(i_lee + 5)
        plt.imshow(
            patterns[i_lee + 1019, :, :], cmap="viridis", origin="lower"
        )  # cmap='viridis', origin='lower'interpolation='none'
        plt.colorbar()
        plt.title("leehol {}".format(i_lee + 1019))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(0)

    plt.show()
