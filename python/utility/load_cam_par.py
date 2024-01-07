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
import os
import sys
import json
import numpy as np
import logging


def load_cam_par(json_file):
    if not os.path.exists(json_file):
        logging.error("calibration file %s does not exist!" % json_file)
        sys.exit(0)

    with open(json_file) as f:
        cam_par = json.load(f)
        k_inv = cam_par["KMatrixInverse"]
        k_inv = np.array(k_inv).reshape(3, 3).astype(float)

        t_vector = cam_par["TVector"]
        t_vector = np.array(t_vector).reshape(3, 1).astype(float)

        r_transpose = cam_par["RMatrixTranspose"]
        r_transpose = np.array(r_transpose).reshape(3, 3).astype(float)

        matrix_geo = cam_par["GMatrix"]
        matrix_geo = np.array(matrix_geo).reshape(4, 4).astype(float)

        c = cam_par["C"]
        ax = cam_par["Afx"]
        ay = cam_par["Afy"]
        b = cam_par["B"]

        return k_inv, t_vector, r_transpose, matrix_geo, c, ax, ay, b

