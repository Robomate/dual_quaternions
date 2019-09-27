#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
#=======================================================================
# Purpose: Rotation by Quaternions
#          Translation by Vector Addition (most efficient solution)
# Version: 08/2019 Roboball
# Links: http://slideplayer.com/slide/5157731/ 
#        https://www.vcalc.com/wiki/vCalc/V3+-+Vector+Rotation
#=======================================================================
'''
import numpy as np

def get_unit_quaternion_rotation(rotation_angle, rotation_vector):
    imag = np.sin(np.deg2rad(rotation_angle/2)) * rotation_vector
    return np.append(np.array([np.cos(np.deg2rad(rotation_angle/2))]),imag)

def get_quaternion_rotation_matrix(unit_quaternion):
    # rotation matrix dervied from quaternion multiplication
    q = unit_quaternion
    q_pow = np.square(q) 
    rot_mat = np.array([[1-2*q_pow[2]-2*q_pow[3], 2*q[1]*q[2]-2*q[0]*q[3], 2*q[1]*q[3]+2*q[0]*q[2], 0],
                        [2*q[1]*q[2]+2*q[0]*q[3], 1-2*q_pow[1]-2*q_pow[3], 2*q[2]*q[3]-2*q[0]*q[1], 0],
                        [2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 1-2*q_pow[1]-2*q_pow[2], 0],
                        [0, 0, 0, 1], ])
    return rot_mat

def apply_quaternion_rotation(input_vector, rotation_matrix):
    point = np.append(input_vector,np.array([0]))[np.newaxis].T
    return np.squeeze(np.dot(rotation_matrix , point)[0:3])

def apply_translation(input_vector, translation_vector):
    # apply translation by vector addition
    return input_vector + translation_vector


if __name__ == '__main__':
    # Inputs
    input_vector = np.array([3,4,5]) # input: original point
    print('\nInput Point: ', input_vector,'\n') 

    #########################################
    # Rotation by Quaternions
    #########################################
    print('================== Rotation ========================')
    rotation_vector = np.array([1,0,0]) # rotation vector
    print('Rotation Axis: ', rotation_vector) 
    rotation_angle = 180 # in degrees
    print('Rotation Angle: ', rotation_angle) 
    
    # (1) get unit quaternion 
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation)
    # (2) get rotation matrix
    rotation_matrix = get_quaternion_rotation_matrix(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix: \n', rotation_matrix)
    # (3) apply rotation to input vector
    output_vector = apply_quaternion_rotation(input_vector, rotation_matrix)
    print('Output Rotation: ', output_vector,'\n')

    ####################################################
    # Most Efficient Solution: 
    # Translation by Vector Addition 
    ####################################################
    print('================== Translation =====================')
    translation_vector = np.array([4,2,6]) # input: original point
    print('Translate by: ', translation_vector) 
    result_rt = apply_translation(output_vector, translation_vector)
    print('Output Translation: ', result_rt,'\n')

   



