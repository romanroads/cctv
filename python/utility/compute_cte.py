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
import logging
from scipy.optimize import curve_fit
import scipy.optimize as optimize


def compute_cte(meas, gt, meas_image_loc, to_fit=True):
    """
    you can select to fit the GPS trace with a polynomial function or not
    :param meas:
    :param gt:
    :param meas_image_loc:
    :param to_fit:
    :return:
    """
    ctes = []

    if to_fit:
        fitfunc = lambda p, x: (p[0] * x) + (p[1] * x ** 2) + (p[2] * x ** 3) + (p[3] * x ** 4) + (p[4] * x ** 5) + p[5]
        residuals = lambda a, x, y: y - fitfunc(a, x)

        x, y = gt[:, 0], gt[:, 1]

        init = [1.0, 1.1, 1.2, 1.5, 1, 1]
        popt, cov, infodict, mesg, ier = optimize.leastsq(residuals, init, args=(x, y), full_output=True)

        x_fit = np.arange(min(x), max(x), 1)
        y_fit = fitfunc(popt, x_fit)

        # Note: use fitted curve as the ground truth instead of the raw GPS trace data
        gt = np.array([(x_fit[i], y_fit[i]) for i in range(len(x_fit))])
    else:
        x_fit = gt[:, 0]
        y_fit = gt[:, 1]

    x_meas = meas[:, 0]
    y_meas = meas[:, 1]

    for i in range(meas.shape[0]):
        pos_meas = meas[i]
        min_dist = np.finfo(float).max
        min_index = None
        num_overshoot = 0

        # Note: for each meas point, we search for closest gt point
        for j in range(gt.shape[0]):
            pos_gt = gt[j]
            vec = pos_meas - pos_gt
            dist = np.linalg.norm(vec)
            if dist <= min_dist:
                min_dist = dist
                min_index = j
            else:
                num_overshoot += 1
                if num_overshoot > 10:
                    break

        if min_index <= 0:
            o1 = gt[min_index + 1]
            o2 = gt[min_index]
            o_min = o2
        else:
            o1 = gt[min_index]
            o2 = gt[min_index - 1]
            o_min = o1

        heading = o1 - o2
        heading_norm = np.linalg.norm(heading)
        if heading_norm > 0:
            heading = heading / heading_norm

        # Linear Path Interpolation (LPI) for Cross-track-error (CTE)
        x1, y1 = o1[0], o1[1]
        x2, y2 = o2[0], o2[1]
        delta_x = x2 - x1

        # Vehicle center
        x_c, y_c = pos_meas[0], pos_meas[1]

        if np.abs(delta_x) > 0:
            a = (y2 - y1) / delta_x
            b = -1.
            c = y1 - x1 * ((y2 - y1) / (x2 - x1))

            # Minimum perpendicular distance point
            x_m = (b * (b * x_c - a * y_c) - a * c) / (a ** 2 + b ** 2)
            y_m = (a * (-b * x_c + a * y_c) - b * c) / (a ** 2 + b ** 2)
        else:
            x_m, y_m = o_min[0], o_min[1]

        x_xte = x_m
        y_xte = y_m

        cte_vector = np.array([x_xte, y_xte]) - pos_meas
        cte = np.linalg.norm(cte_vector)
        cte_vector_norm = cte_vector / cte
        cte_vector_norm = np.array([cte_vector_norm[0], cte_vector_norm[1], 0])
        heading = np.array([heading[0], heading[1], 0])
        cross_prod_vector = np.cross(heading, cte_vector_norm)

        z = cross_prod_vector[2]
        sign = np.sign(z)

        cte_signed = cte * sign

        # Note: sign is positive if meas trace is on the right-hand-side on gt / ref trace
        # when looking at the direction of movement
        ctes.append((cte_signed, meas_image_loc[i], meas[i]))

    return ctes, x_fit, y_fit, x_meas, y_meas


def compute_time_dependent_error(meas_space, meas_time, gt_space, gt_time, meas_image_loc):

    delta_t_gt = gt_time[1] - gt_time[0]
    logging.info("compute_cte: ground truth trace was recorded at period of %.3f ms" % delta_t_gt)

    # Note: let's search for a little window near the timestamp match point to account for the systematic error
    # caused by temporal synchronization
    windows_size_ms = 100  # in seconds
    windows_size = int(windows_size_ms // delta_t_gt)

    # Note: exception handling in case you couldn't find a heading vector
    heading_default = meas_space[-1] - meas_space[0]
    heading_norm = np.linalg.norm(heading_default)
    if heading_norm > 0:
        heading_default = heading_default / heading_norm
    else:
        heading_default = np.array([1, 0])

    delta_ds, delta_lat_ds, delta_lon_ds = [], [], []
    for i in range(meas_time.shape[0]):
        t_meas = meas_time[i]

        # Note: we require an exact match in timestamp for now
        index_matched = np.where(gt_time == t_meas)[0]
        if len(index_matched) <= 0:
            continue

        pos_meas = meas_space[i]

        if i > 0:
            pos_meas_prev = meas_space[i - 1]
            heading = pos_meas - pos_meas_prev
        else:
            pos_meas_next = meas_space[i + 1]
            heading = pos_meas_next - pos_meas

        heading_norm = np.linalg.norm(heading)
        if heading_norm > 0:
            heading = heading / heading_norm
        else:
            heading = heading_default

        index_matched = int(index_matched[0])
        t_gt = gt_time[index_matched]
        pos_gt = gt_space[index_matched]

        vect = pos_gt - pos_meas
        dist = np.linalg.norm(vect)

        index_approximated = index_matched - windows_size
        while index_approximated < index_matched + windows_size:
            if index_approximated >= 0 or index_approximated < gt_time.shape[0]:
                pos_gt_approx = gt_space[index_approximated]
                vect_approx = pos_gt_approx - pos_meas
                dist_approx = np.linalg.norm(vect_approx)
                if dist_approx < dist:
                    pos_gt = pos_gt_approx
                    vect = vect_approx
                    dist = dist_approx
            index_approximated += 1

        vector_3d = np.array([vect[0], vect[1], 0])
        heading_3d = np.array([heading[0], heading[1], 0])
        cross_prod_vector = np.cross(heading_3d, vector_3d)
        sign_lat = np.sign(cross_prod_vector[2])

        inner_prod = np.dot(heading, vect)
        sign_lon = np.sign(inner_prod)

        # Note: sign is positive if meas is on the right-hand-side on gt / ref point
        # when looking at the direction of movement
        delta_lat_d = sign_lat * np.linalg.norm(cross_prod_vector)

        # Note: sign is positive if meas is in front of or at downstream location of gt / ref point
        # when looking at the direction of movement
        delta_lon_d = sign_lon * np.abs(inner_prod)

        delta_ds.append((dist * sign_lon, meas_image_loc[i], meas_space[i]))
        delta_lat_ds.append((delta_lat_d, meas_image_loc[i], meas_space[i]))
        delta_lon_ds.append((delta_lon_d, meas_image_loc[i], meas_space[i]))

    return delta_ds, delta_lat_ds, delta_lon_ds
