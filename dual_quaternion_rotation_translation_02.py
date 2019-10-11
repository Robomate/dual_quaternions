#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
#===========================================================================================
# Purpose: Rotation and Translation by DualQuaternions
# Version: 08/2019 Roboball
# Links: http://slideplayer.com/slide/5157731/ 
#        http://ekunzeweb.de/PAPERS/Transformationen%20mit%20Dualen%20Quaternionen.pdf
#        https://www.vcalc.com/wiki/vCalc/V3+-+Vector+Rotation
#===========================================================================================
'''
import numpy as np

def get_unit_quaternion_rotation(rotation_angle, rotation_vector):
    imag = np.sin(np.deg2rad(rotation_angle/2)) * rotation_vector
    return np.append(np.array([np.cos(np.deg2rad(rotation_angle/2))]),imag)

def get_unit_quaternion_translation(translation_vector):
    imag = 0.5 * translation_vector
    return np.append(np.array([0]),imag)

def matrix_vector_mul(matrix, vector):
    return np.dot(matrix, vector)

def get_quaternion_conjugate(input_quaternion):
    # ordered by:  w + imag (i,j,k)
    a = input_quaternion
    return np.array([a[0], -a[1], -a[2], -a[3]])

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

def get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix):
    matrix_01 = np.append(unit_quaternion_rotation_matrix, np.zeros((4, 4)), axis=1)
    matrix_02 = np.append(unit_quaternion_translation_matrix, unit_quaternion_rotation_matrix, axis=1)
    matrix = np.append(matrix_01, matrix_02, axis=0)
    return matrix

def dual_quaternion_calculation(input_vector, rotation_angle,rotation_vector,translation_vector):
    # input Dual Quaternion
    input_dual_quaternion = np.zeros(8)
    input_dual_quaternion[0] = 1
    input_dual_quaternion[5:] = input_vector
    print('Input Dual Quaternion: ', input_dual_quaternion,'\n') 

    # 1# mul ###################################################################################################
    # (2a) get unit quaternions
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation)
    unit_quaternion_translation = get_unit_quaternion_translation(translation_vector)   
    print('Unit Quaternion Translation: ', unit_quaternion_translation,'\n')
     # (2b) get unit quaternion conjugates 
    unit_quaternion_rotation_conjugate  = get_quaternion_conjugate(unit_quaternion_rotation)
    print('Unit Quaternion Rotation Conjugate: ', unit_quaternion_rotation_conjugate)
    # positive due to [p-eq], [(p0-p1-p2-p3)-e(q0-q1-q2-q3)]
    print('Unit Quaternion Translation Conjugate: ', unit_quaternion_translation,'\n') 
    # (2c) build DQ conjugate
    dual_quaternion_conjugate  = np.append(unit_quaternion_rotation_conjugate,unit_quaternion_translation)
    print('Dual Quaternion Conjugate: ', dual_quaternion_conjugate,'\n')

    # (3) get rotation matrix
    unit_quaternion_rotation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix: \n', unit_quaternion_rotation_matrix,'\n')

    # (4) get translation matrix
    unit_quaternion_translation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_translation) 
    with np.printoptions(precision=3, suppress=True):
        print('Translation Matrix: \n', unit_quaternion_translation_matrix,'\n')

    # (5) get DQ matrix
    dq_matrix_01 = get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)
    with np.printoptions(precision=3, suppress=True):
        print('DQ Matrix 01: \n', dq_matrix_01,'\n')

    # (6) Apply Dual Quaternion Matrix Muliplication
    output_vector_01 = matrix_vector_mul(dq_matrix_01, input_dual_quaternion)
    print('DQ output_vector_01: ', output_vector_01,'\n')

    # 2# mul ###################################################################################################
    rotation_vector_02 = output_vector_01[0:4]
    print('Rotation Vec 02: ', rotation_vector_02) 
    translation_vector_02 = output_vector_01[4:]
    print('Translate Vec 02: ', translation_vector_02,'\n')

    # (3) get rotation matrix
    rotation_matrix_02 = get_matrix_for_quaternion_mul(rotation_vector_02)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix 02: \n', rotation_matrix_02,'\n')
    
    # (4) get translation matrix
    translation_matrix_02 = get_matrix_for_quaternion_mul(translation_vector_02) 
    with np.printoptions(precision=3, suppress=True):
        print('Translation Matrix 02: \n', translation_matrix_02,'\n')
    
    # (5) get DQ matrix
    dq_matrix_02 = get_matrix_for_dual_quaternion_mul(rotation_matrix_02, translation_matrix_02)
    with np.printoptions(precision=3, suppress=True):
        print('DQ Matrix 02: \n', dq_matrix_02,'\n')

    # (6) Apply Dual Quaternion Matrix Muliplication
    output_vector_02 = matrix_vector_mul(dq_matrix_02, dual_quaternion_conjugate)
    print('DQ output_vector_02: ', output_vector_02,'\n')

    # (7) Result
    return output_vector_02[5:]
    


if __name__ == '__main__':
    # Input
    print('================== Dual Quaternions ========================')
    input_vector = np.array([3,4,5]) # input: original point
    print('\nInput Point: ', input_vector) 

    ########################################################
    # Dual-Quaternions: Pure Rotation
    ########################################################
    print('\n######################################################')
    print('Dual-Quaternions: Pure Rotation')
    print('######################################################\n')

    # (1) input vectors
    rotation_angle = 180 # in degrees
    rotation_vector = np.array([1,0,0]) # rotation vector
    print('Rotate by: ', rotation_angle,'Degrees, Rotation Axis: ', rotation_vector) 
    translation_vector = np.array([0,0,0]) # translate input by translation vector
    print('Translate by: ', translation_vector,'\n') 
    # (2) result
    result = dual_quaternion_calculation(input_vector, rotation_angle,rotation_vector,translation_vector)
    print('Result Pure Translation: ', result,'\n')

    ########################################################
    # Dual-Quaternions: Pure Translation
    ########################################################
    print('\n######################################################')
    print('Dual-Quaternions: Pure Translation')
    print('######################################################\n')

    # (1) input vectors
    rotation_angle = 0 # in degrees
    rotation_vector = np.array([0,0,0]) # rotation vector
    print('Rotate by: ', rotation_angle,'Degrees, Rotation Axis: ', rotation_vector) 
    translation_vector = np.array([4,2,6]) # translate input by translation vector
    print('Translate by: ', translation_vector,'\n') 
    # (2) result
    result = dual_quaternion_calculation(input_vector, rotation_angle,rotation_vector,translation_vector)
    print('Result Pure Translation: ', result,'\n')

    # ########################################################
    # # Dual-Quaternions: Rotation followed by Translation
    # ########################################################
    # print('\n######################################################')
    # print('Dual-Quaternions: Rotation followed by Translation')
    # print('######################################################\n')

    # # (1) input vectors
    # rotation_angle = 180 # in degrees
    # rotation_vector = np.array([1,0,0]) # rotation vector
    # print('Rotate by: ', rotation_angle,'Degrees, Rotation Axis: ', rotation_vector) 
    # translation_vector = np.array([4,2,6]) # translate input by translation vector
    # print('Translate by: ', translation_vector,'\n') 
    # # (2) result
    # result = dual_quaternion_calculation(input_vector, rotation_angle,rotation_vector,translation_vector)
    # print('Result Rotation followed by Translation: ', result,'\n')

    ########################################################
    # Dual-Quaternions: Translation followed by Rotation 
    ########################################################
    print('\n######################################################')
    print('Dual-Quaternions: Translation followed by Rotation')
    print('######################################################\n')

    # (1) input vectors
    rotation_angle = 180 # in degrees
    rotation_vector = np.array([1,0,0]) # rotation vector
    print('Rotate by: ', rotation_angle,'Degrees, Rotation Axis: ', rotation_vector) 
    translation_vector = np.array([4,2,6]) # translate input by translation vector
    print('Translate by: ', translation_vector,'\n') 
    # (2) result
    result = dual_quaternion_calculation(input_vector, rotation_angle,rotation_vector,translation_vector)
    print('Result Translation followed by Rotation: ', result,'\n')



    
