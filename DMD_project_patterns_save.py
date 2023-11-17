#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2023/8/2 16:39
@author: Zhen Cheng
"""
from DMD_holography import *
import matplotlib.pyplot as plt
import code_tools

logger = code_tools.get_logger("DMD_project_patterns_save")

logger.debug('DMD_project_patterns_save debug mode ON.')

# Set parameters
axial_method = 2 # 1: coaxial 2: offaxial
DMDwidth = 1920
DMDheight = 1080
holo_method = 1  # 1: LeePattern, 2: BoydPattern
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

dmdpat = PatternProject(DMDwidth, DMDheight)

superpixel = dmdpat.getparameter(hadamardSize, referenceFraction)

# Generate patterns
block_size_loop_put = 1024
phase_array = dmdpat.get_hadamardBasis_phaseArrray(hadamardSize)
for i_put in range(0, hadamardSize**2, block_size_loop_put):
    logger.info("------------loading put project number {}-----------".format(i_put))
    start_put = i_put
    end_put = min(i_put + block_size_loop_put, hadamardSize**2)
    current_phase_array = phase_array[start_put:end_put]
    if axial_method == 1:
        current_phase_array = dmdpat.getGratingProject_phase_coaxial(
            superpixel, current_phase_array, phase_4, holo_method,
            amplitude=amplitude, alpha=alpha, x0=x0, rot=rot)
        file_name = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_coaxial_{start_put}_{end_put}_methold{holo_method}.npy'
        np.save('./measure_patterns/' + file_name, current_phase_array)
        logger.info(f"Save {file_name} to measure_patterns")
        code_tools.show_images(current_phase_array[0].reshape((1,) + current_phase_array[0].shape), 1, 1, scale=15)

    elif axial_method == 2:
        current_phase_array = dmdpat.getGratingProject_phase_offaxial(
            superpixel, current_phase_array, phase_4, holo_method,
            amplitude=amplitude, alpha=alpha, x0=x0, rot=rot)
        file_name = f'{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_offaxial_{start_put}_{end_put}_methold{holo_method}.npy'
        np.save('./measure_patterns/' + file_name, current_phase_array)
        logger.info(f"Save {file_name} to measure_patterns")
        code_tools.show_images(phase_array[0].reshape((1,) + phase_array[0].shape), 1, 1, scale=15)
        code_tools.show_images(current_phase_array[0].reshape((1,) + current_phase_array[0].shape), 1, 1, scale=15)

    else:
        logger.error("Must choose axial_methold 1: coaxial or 2: offaxial")
        raise ValueError("Must choose axial_methold 1: coaxial or 2: offaxial")

# plt.savefig(f'./measure_img/{hadamardSize}hadamard_{DMDwidth}_{DMDheight}_methold{holo_method}.svg', format='svg')
plt.show()

