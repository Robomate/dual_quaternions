#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
#=======================================================================
# Purpose: Rotations by Quaternions
# Version: 08/2019 Roboball (MattK.)
# Links: http://slideplayer.com/slide/5157731/ 
#=======================================================================
'''
import numpy as np
# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

def get_unit_quaternion_rotation(rotation_angle, rotation_vector):
    imag = np.sin(np.deg2rad(rotation_angle/2)) * rotation_vector
    return np.append(np.array([np.cos(np.deg2rad(rotation_angle/2))]),imag)

def get_unit_quaternion_translation(translation_vector):
    imag = 0.5 * translation_vector
    return np.append(np.array([1]),imag)

def get_quaternion_conjugate(input_quaternion):
    # ordered by:  w + imag (i,j,k)
    a = input_quaternion
    return np.array([a[0], -a[1], -a[2], -a[3]])
    

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
    print('\nInput Quat ', point ,'\n') 
    # point2 = np.dot(rotation_matrix , point)
    return np.squeeze(np.dot(rotation_matrix , point)[0:3])

def apply_translation(input_vector, translation_vector):
    # apply translation by vector addition
    return input_vector + translation_vector



# def quaternion_mul(input_vector_a,input_vector_c):
#     # https://www.youtube.com/watch?v=3Ki14CsP_9k
#     a = input_vector_a
#     mat = np.array([ 
#                     [ a[0], -a[3],  a[2], a[1]],
#                     [ a[3],  a[0], -a[1], a[2]],
#                     [-a[2],  a[1],  a[0], a[3]],
#                     [-a[1], -a[2], -a[3], a[0]], 
#                     ])
#     return np.dot(mat, input_vector_c)

# def quaternion_mul(input_vector_a,input_vector_c):
#     # https://www.youtube.com/watch?v=3Ki14CsP_9k
#     # normale reihenfolge w,(jkl)
#     a = input_vector_a
#     mat = np.array([ 
#                     [ a[0], -a[1], -a[2], -a[3]],
#                     [ a[1],  a[0], -a[3],  a[2]],
#                     [ a[2],  a[3],  a[0], -a[1]],
#                     [ a[3], -a[2],  a[1], a[0]], 
#                     ])
#     return np.dot(mat, input_vector_c)



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

# def get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix):
#     # matrix = np.concatenate((unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)), axis=2)
#     matrix_01 = np.append(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix, axis=1)
#     matrix_02 = np.zeros((4, 4))
#     matrix_02 = np.append(matrix_02, unit_quaternion_rotation_matrix, axis=1)
#     matrix = np.append(matrix_01, matrix_02, axis=0)
#     # matrix = np.array([ quaternion_matrix_rotation , 
#     return matrix


def get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix):
    # matrix = np.concatenate((unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)), axis=2)
    matrix_01 = np.append(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix, axis=1)
    matrix_02 = np.zeros((4, 4))
    matrix_02 = np.append(matrix_02, unit_quaternion_rotation_matrix, axis=1)
    matrix = np.append(matrix_01, matrix_02, axis=0)
    # matrix = np.array([ quaternion_matrix_rotation , 
    return matrix

def get_matrix_for_dual_quaternion_mul_02(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix):
    # matrix = np.concatenate((unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)), axis=2)
    matrix_01 = np.append(unit_quaternion_rotation_matrix, np.zeros((4, 4)), axis=1)
    matrix_02 = np.append(unit_quaternion_translation_matrix, unit_quaternion_rotation_matrix, axis=1)
    matrix = np.append(matrix_01, matrix_02, axis=0)
    # matrix = np.array([ quaternion_matrix_rotation , 
    return matrix



# def apply_quaternion_matrix(point_input, rotation_angle_vector):
#     p = np.append(np.array([0]),point_input)[np.newaxis].T # original point [translation=0,x=1,y=0,z=0]
#     p_pow = np.square(p) # original point squared
#     # rotation matrix dervied from quaternion multiplication
#     rot_matrix = np.array([ [1-2*p_pow[2]-2*p_pow[3], 2*p[1]*p[2]+2*p[0]*p[3], 2*p[1]*p[3]-2*p[0]*p[2], 0],
#                             [2*p[1]*p[2]-2*p[0]*p[3], 1-2*p_pow[1]-2*p_pow[3], 2*p[2]*p[3]+2*p[0]*p[1], 0],
#                             [2*p[1]*p[3]+2*p[0]*p[2], 2*p[2]*p[3]-2*p[0]*p[1], 1-2*p_pow[1]-2*p_pow[2], 0],
#                             [0, 0, 0, 1], ])
#     p_result = np.dot(rot_matrix , p)
#     return p_result[0:3]


if __name__ == '__main__':

    #########################################
    # Quaternions: Pure Rotations
    #########################################
    # Example: 1
    # Check online here: https://www.vcalc.com/wiki/vCalc/V3+-+Vector+Rotation
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/examples/index.htm

    input_vector = np.array([3,4,5]) # input: original point
    print('\nInput Point: ', input_vector,'\n') 
    rotation_vector = np.array([1,0,0]) # rotation vector
    rotation_angle= 180 # in degrees

    # (1) get unit quaternion 
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation,'\n')
    # (2) get rotation matrix
    rotation_matrix = get_quaternion_rotation_matrix(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix: \n', rotation_matrix,'\n')
    # (3) apply rotation to input vector
    output_vector = apply_quaternion_rotation(input_vector, rotation_matrix)
    print('Output Point: ', output_vector,'\n')

    ####################################################
    # Most Efficient Solution: 
    # Rotation: by Quaternions (about the origin)
    # Translation: by Vector Addition 
    ####################################################
    translation_vector = np.array([4,2,6]) # input: original point
    result_rt = apply_translation(output_vector, translation_vector)
    print('Result_rot_trans: ', result_rt,'\n')

    ###############################################
    # Alternative 2nd efficient Solution: 
    # Dual-Quaternions: Translation + Rotation
    ###############################################

    # https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/other/dualQuaternion/index.htm
    # Links: https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    # http://web.cs.iastate.edu/~cs577/handouts/dual-quaternion.pdf

    # https://www.euclideanspace.com/maths/geometry/affine/matrix4x4/index.htm

    ###############################################
    # Dual-Quaternions: Pure Translation
    ###############################################
    print('\n###############################################')
    print('Dual-Quaternions: Pure Translation')
    print('###############################################\n')

    # (1) input vectors
    rotation_angle = 0 # in degrees
    rotation_vector = np.array([0,0,0]) # rotation vector
    translation_vector = np.array([4,2,6]) # translate input by translation vector

    # (2) get unit quaternions
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation,'\n')
    unit_quaternion_translation = get_unit_quaternion_translation(translation_vector)   ##### noch falsch !!! erste stelle muss NUll sien
    print('Unit Quaternion Translation: ', unit_quaternion_translation,'\n')

    # (3) get rotation matrix
    unit_quaternion_rotation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix: \n', unit_quaternion_rotation_matrix,'\n')

    # (4) get translation matrix
    unit_quaternion_translation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_translation) 
    with np.printoptions(precision=3, suppress=True):
        print('Translation Matrix: \n', unit_quaternion_translation_matrix,'\n')


    
    # (5) get translation matrix
    dq_matrix_01 = get_matrix_for_dual_quaternion_mul_02(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)
    with np.printoptions(precision=3, suppress=True):
        print('DQ Matrix 01: \n', dq_matrix_01,'\n')




    # get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)



    


   
    # # (3) apply rotation to input vector
    # output_vector = apply_quaternion_rotation(input_vector, translation_matrix)
    # print('Output Point: ', output_vector,'\n')

    ###############################################
    # Dual-Quaternions: Pure Rotation
    ###############################################
    print('\n###############################################')
    print('Dual-Quaternions: Pure Rotation')
    print('###############################################\n')

    ##see quaternion rotation

    ##or with 8x8 matrices

    unit_quaternion_rotation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_rotation)
    with np.printoptions(precision=3, suppress=True):
        print('Rotation Matrix: \n', unit_quaternion_rotation_matrix,'\n')

    unit_quaternion_translation_matrix = np.zeros([4,4])
    zero_matrix4 = np.zeros([4,4])
    print('Unit Dual Quaternion Translation Matrix:\n', unit_quaternion_translation_matrix,'\n')

    matrix_top = np.append(zero_matrix4, unit_quaternion_rotation_matrix, axis=1)
    matrix_down = np.append(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix, axis=1)
    with np.printoptions(precision=3, suppress=True):
        print('Matrix: \n', matrix_down,'\n')

    # matrix = np.append(matrix_01, matrix_02, axis=0)



    # (3a)Dual Quaternion Matrix [8x8] 
    # unit_dual_quaternion_matrix = get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)
    # print('Unit Dual Quaternion Matrix:\n', unit_dual_quaternion_matrix,'\n')


    ###############################################
    # Dual-Quaternions: Pure Rotation
    ###############################################
    print('\n###############################################')
    print('Dual-Quaternions: Pure Translation')
    print('###############################################\n')

    

    #######################################################
    # Dual-Quaternions: Rotation followed by Translation
    #######################################################
    print('\n#####################################################')
    print('Dual-Quaternions: Rotation followed by Translation')
    print('#####################################################\n')

    # (1a) rotation: get unit quaternion 
    unit_quaternion_rotation = get_unit_quaternion_rotation(rotation_angle, rotation_vector)
    print('Unit Quaternion Rotation: ', unit_quaternion_rotation,'\n')
    unit_quaternion_rotation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_rotation)
    print('Unit Quaternion Rotation Matrix:\n', unit_quaternion_rotation_matrix,'\n')

    # (1b) rotation: get unit quaternion conjugate
    unit_quaternion_rotation_conjugate  = get_quaternion_conjugate(unit_quaternion_rotation)
    print('Unit Quaternion Rotation Conjugate: ', unit_quaternion_rotation_conjugate,'\n')
    unit_quaternion_rotation_matrix_conjugate = get_matrix_for_quaternion_mul(unit_quaternion_rotation_conjugate)
    print('Unit Quaternion Rotation Matrix Conjugate:\n', unit_quaternion_rotation_matrix_conjugate,'\n')

    # (2a) translation: get unit quaternion 
    unit_quaternion_translation = get_unit_quaternion_translation(translation_vector)
    print('Unit Quaternion Translation: ', unit_quaternion_translation,'\n')
    unit_quaternion_translation_matrix = get_matrix_for_quaternion_mul(unit_quaternion_translation)
    print('Unit Quaternion Translation Matrix:\n', unit_quaternion_translation_matrix,'\n')

    # (2b) translation: get unit quaternion conjugate
    unit_quaternion_translation_conjugate  = get_quaternion_conjugate(unit_quaternion_translation)
    print('Unit Quaternion Translation Conjugate: ', unit_quaternion_translation_conjugate ,'\n')
    unit_quaternion_translation_matrix_conjugate = get_matrix_for_quaternion_mul(unit_quaternion_translation_conjugate)
    print('Unit Quaternion Translation Matrix Conjugate:\n', unit_quaternion_translation_matrix_conjugate,'\n')

    # (3a)Dual Quaternion Matrix [8x8] 
    unit_dual_quaternion_matrix = get_matrix_for_dual_quaternion_mul(unit_quaternion_rotation_matrix, unit_quaternion_translation_matrix)
    print('Unit Dual Quaternion Matrix:\n', unit_dual_quaternion_matrix,'\n')




    # Wenn wir das ueberhaupt brauchen?????:
    # (3b)Dual Quaternion Matrix Conjugate[8x8] 

    # (4a)Apply Dual Quaternion Matrix Muliplication
    input_dual_quaternion = np.ones(8)
    output_vector_01 = matrix_vector_mul(unit_dual_quaternion_matrix, input_dual_quaternion)
    print('Matrix output_vector_01: ', output_vector_01,'\n')
    










    #######################################################
    # Dual-Quaternions: Translation followed by Rotation  
    #######################################################
    print('\n#####################################################')
    print('Dual-Quaternions: Translation followed by Rotation')
    print('#####################################################\n')


   



    #######################################################
    # Quaternion Multiplication 
    #######################################################

    # apply 2 quaternion multiplications
    # eqn: P2 = q * P1 * q_conjugate

    # Rotation
    # print('\n#####################################################')
    # input_vector_a = np.array([1, 3, 4, 5]) 

    # dq_unit = unit_quaternion_rotation
    # print('quat_mul: ', dq_unit,'\n')
    # dq_unit_conjugate = get_quaternion_conjugate(dq_unit)
    # print('conjugate: ', dq_unit_conjugate,'\n')

    # quat_mul_02 = quaternion_mul(dq_unit, input_vector_a)
    # print('quat_mul: ', quat_mul_02,'\n')

    # quat_mul_03 = quaternion_mul(quat_mul_02, dq_unit_conjugate)
    # print('quat_mul: ', quat_mul_03,'\n')

    # # Translation
    # print('\n#####################################################')
    # dq_unit_trans = unit_quaternion_translation




