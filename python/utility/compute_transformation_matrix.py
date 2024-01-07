# Copyright 2019 - 2021 The ROMAN ROADS Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np


def compute_transformation_matrix(k_inv, t_vector, r_transpose, matrix_geo):
    matrix_a = np.zeros((4, 4)).astype(float)
    matrix_a_3x3 = np.matmul(r_transpose, k_inv)
    matrix_a[:3, :3] = matrix_a_3x3
    matrix_a[3, :] = [0, 0, 0, 1.]
    matrix_a = np.matmul(matrix_geo, matrix_a)

    vector_b = np.zeros((4, 1))
    vector_b_3x1 = np.matmul(r_transpose, t_vector)
    vector_b[:3, :1] = vector_b_3x1
    vector_b = np.matmul(matrix_geo, vector_b)

    return matrix_a, vector_b


def compute_estimate_local_z_component(f_x_center, f_y_center, c, ax, ay, b):
    return c / (ax * f_x_center + ay * f_y_center + b)

