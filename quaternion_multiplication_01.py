#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
#=======================================================================
# Purpose: Unit Quaternion Multiplication (for Rotations)
# Version: 08/2019 Roboball
# Links: http://slideplayer.com/slide/5157731/ 
#        https://www.vcalc.com/wiki/vCalc/V3+-+Vector+Rotation
#=======================================================================
'''
import numpy as np

def get_quaternion_conjugate(input_quaternion):
    # ordered by:  w + imag (i,j,k)
    a = input_quaternion
    return np.array([a[0], -a[1], -a[2], -a[3]])

def get_unit_quaternion_rotation(rotation_angle, rotation_vector):
    imag = np.sin(np.deg2rad(rotation_angle/2)) * rotation_vector
    return np.append(np.array([np.cos(np.deg2rad(rotation_angle/2))]),imag)

def get_matrix_for_quaternion_mul(input_quaternion):
    # ordered by:  w + imag (i,j,k)
    a = input_quaternion
    matrix = np.array([ 
                    [ a[0], -a[1], -a[2], -a[3]],
                    [ a[1],  a[0], -a[3],  a[2]],
                    [ a[2],  a[3],  a[0], -a[1]],
                    [ a[3], -a[2],  a[1], a[0]], 
                    ])
    return matrix

def matrix_vector_mul(matrix, vector):
    return np.dot(matrix, vector)



if __name__ == '__main__':
    # Inputs
    input_vector = np.array([3,4,5]) # input: original point
    print('\nInput Point: ', input_vector) 

    #########################################
    # Rotation by Quaternions
    #########################################
    print('================== Rotation ========================')
    rotation_vector = np.array([1,0,0]) # rotation vector
    print('Rotation Axis: ', rotation_vector) 
    rotation_angle = 180 # in degrees
    print('Rotation Angle: ', rotation_angle) 

    ############################################################################################
    # (1a) get unit quaternion
    input_quaternion = np.insert(input_vector, 0, 1) # vector, pos, value
    print('Input Quaternion: ', input_quaternion ,'\n') 
    # (1b) get unit quaternion
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation)
    # (1c) get unit quaternion conjugate
    unit_quaternion_rotation_conjugate  = get_quaternion_conjugate(unit_quaternion_rotation)
    print('Unit Quaternion Rotation Conjugate: ', unit_quaternion_rotation_conjugate,'\n')
    ############################################################################################
    # (2a) get matrix 01
    unit_quaternion__matrix_01 = get_matrix_for_quaternion_mul(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix 01: \n', unit_quaternion__matrix_01)
    # (2b) matmul 01
    quaternion_01 = matrix_vector_mul(unit_quaternion__matrix_01, input_quaternion)
    print('Quaternion_01 : ', quaternion_01 ,'\n')
    ############################################################################################
    # (3a) get matrix 02
    unit_quaternion__matrix_02 = get_matrix_for_quaternion_mul(quaternion_01)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix 02: \n', unit_quaternion__matrix_02)
    # (3b) matmul 02
    quaternion_02 = matrix_vector_mul(unit_quaternion__matrix_02, unit_quaternion_rotation_conjugate)
    print('Quaternion_02 : ', quaternion_02 ,'\n')

    print('================== Result ========================')
    output_vector = quaternion_02[1:]
    print('Output Result: ', output_vector,'\n')

    
   



