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
import sys
import os
from sqlalchemy import create_engine
import pandas as pd
import cv2
import numpy as np
import logging
from datetime import datetime
import scipy.optimize as optimize
from scipy import interpolate
import imageio

from common.tracklet import Tracklet

CUR_ID = 0


def connect_to_database(user_name, password, endpoint, port, db_name):
    address = 'postgresql://%s:%s@%s:%s/%s' % (user_name, password, endpoint, port, db_name)
    engine = create_engine(address)
    connection = engine.raw_connection()
    cursor = connection.cursor()
    return connection, cursor


def close_connection(conn):
    logging.info("closing connections to SQL database now .....")
    conn.close()


def compute_histogram_bins(nbins, bin_start, bin_end):
    x_start, x_end, x_bins = bin_start, bin_end, nbins
    x_bin_size = (x_end - x_start) * 1. / x_bins

    x_bin_center, x_bin_edge = [], []

    for i in range(x_bins):
        x_bin_center.append(x_start + (i + 0.5) * x_bin_size)

    for i in range(x_bins + 1):
        x_bin_edge.append(x_start + i * 1. * x_bin_size)

    return x_bin_center, x_bin_edge


def compute_the_2d_weighted_histogram(x_start, x_end, x_bins, y_start, y_end, y_bins):
    x_bin_size = (x_end - x_start) * 1. / x_bins
    y_bin_size = (y_end - y_start) * 1. / y_bins

    x_bin_center, y_bin_center = [], []
    x_bin_edge, y_bin_edge = [], []

    for i in range(x_bins):
        x_bin_center.append(x_start + (i + 0.5) * x_bin_size)
    for i in range(y_bins):
        y_bin_center.append(y_start + (i + 0.5) * y_bin_size)

    for i in range(x_bins + 1):
        x_bin_edge.append(x_start + i * 1. * x_bin_size)
    for i in range(y_bins + 1):
        y_bin_edge.append(y_start + i * 1. * y_bin_size)

    return x_bin_edge, y_bin_edge


def load_gps(csv_file, start_step, num_steps):
    df_total = pd.read_csv(csv_file, header=0,
                           skiprows=3,
                           names=["Session", "Time",
                                  "Location mode",
                                  "Longitude", "Latitude",
                                  "Altitude", "Speed", "Orientation", "Pitch angle",
                                  "Roll angle", "Yaw rate", "Acceleration-x", "Acceleration-y",
                                  "Acceleration-z", "Year", "Month", "Day", "Hour", "Minute", "Second", "Millisecond"])
    # Note: this 8 hours correction is due to wrong GPS timezone setup...
    num_hours_offset_GMT = 8

    # Note: GPS data has no timestsamp column, had to construct it from other columns
    df_total["Timestamp"] = df_total.apply(
        lambda x: int(datetime(year=x['Year'], month=x['Month'], day=x['Day'], hour=x["Hour"] + num_hours_offset_GMT,
                               minute=x['Minute'], second=x['Second'],
                               microsecond=x["Millisecond"] * 1000).timestamp() * 1000), axis=1)

    df_total = df_total[["Latitude", "Longitude", "Timestamp"]]

    # TODO need to use inverse transform to get pixel coordinates later
    df_total['Fx'] = 0
    df_total['Fy'] = 0
    df_total['Fw'] = 0
    df_total['Fh'] = 0

    return df_total.iloc[start_step:start_step + num_steps, :]


def load_cam(csv_file, video_file, target_agent_id, auto, window_name, escape_key, width_window, window_start_width,
             window_start_height, ts_start, batch=False):
    global CUR_ID

    df_total = pd.read_csv(csv_file)
    thumbnail_image = None

    # Note: cached tracklets or agents
    tracklets = []

    # Note: cached target vehicle trace
    trace = []

    cap = cv2.VideoCapture(video_file)
    input_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    period_ms = int(1000. / fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_max = frame_count - 1

    logging.info("input video has %s frames, with frame size w (%s) x h (%s) and fps %.1f, frame_max = %s" %
                 (frame_count, input_video_width, input_video_height, fps, frame_max))

    if not batch:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)

    height_window = int(width_window * (input_video_height * 1. / input_video_width))

    if not batch:
        cv2.resizeWindow(window_name, width_window, height_window)
        cv2.moveWindow(window_name, window_start_width, window_start_height)

    frame_num = 0
    CUR_ID = 0

    while cap.isOpened() and frame_num < frame_count and frame_num < frame_max:
        _ret, frame_ori = cap.read()

        if not _ret:
            break

        ts = ts_start + frame_num * period_ms

        if thumbnail_image is None:
            thumbnail_image = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)

        df_f = df_total.loc[df_total.frame_id == frame_num]
        det_ids = list(df_f.agent_id)

        detections = []
        for det_id in det_ids:
            df_a = df_f.loc[df_f.agent_id == det_id]
            lat, lon, alt, fx, fy, fw, fh = float(df_a.latitude), float(df_a.longitude), float(df_a.altitude),\
                                            float(df_a.fx), float(df_a.fy), float(df_a.f_w), float(df_a.f_h)
            detections.append((frame_num, det_id, lat, lon, alt, fx, fy, fw, fh, ts))

        update_existing_and_create_new_tracklets_using_detection(tracklets, detections, fps)
        update_existing_tracklets_using_image_tracking(tracklets, frame_ori, frame_num, target_agent_id, trace)

        if not batch:
            cv2.imshow(window_name, frame_ori)

        if auto:
            wait_time = 1
        else:
            wait_time = 0

        key = cv2.waitKey(wait_time) & 0xFF
        if not batch:
            if key == escape_key or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(window_name)
                cap.release()
                sys.exit(0)
            elif key == ord("q"):
                break
            elif key == ord("n"):
                pass
        else:
            pass

        frame_num += 1

    df = pd.DataFrame(trace, columns=['Latitude', 'Longitude', 'Fx', 'Fy', 'Fw', 'Fh', 'Timestamp'])
    return df, thumbnail_image


def load_cam_generate_gif(csv_file, video_file, target_agent_id, auto, window_name, escape_key, width_window,
                          window_start_width, window_start_height, ts_start, is_cam_gif):
    global CUR_ID

    df_total = pd.read_csv(csv_file)
    thumbnail_image = None

    # Note: cached tracklets or agents
    tracklets = []

    # Note: cached target vehicle trace
    trace = []
    trace_index = 0
    prev_trace_len = -1
    trace_w_filtered_total = []
    trace_h_filtered_total = []
    trace_w_min, trace_w_max = -1, -1
    images_for_gif = []
    ego_images_for_gif = []

    cap = cv2.VideoCapture(video_file)
    input_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    period_ms = int(1000. / fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_max = frame_count - 1

    logging.info("input video has %s frames, with frame size w (%s) x h (%s) and fps %.1f, frame_max = %s" %
                 (frame_count, input_video_width, input_video_height, fps, frame_max))

    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
    height_window = int(width_window * (input_video_height * 1. / input_video_width))
    cv2.resizeWindow(window_name, width_window, height_window)
    cv2.moveWindow(window_name, window_start_width, window_start_height)

    frame_num = 0
    CUR_ID = 0

    while cap.isOpened() and frame_num < frame_count and frame_num < frame_max:
        _ret, frame_ori = cap.read()

        if not _ret:
            break

        ts = ts_start + frame_num * period_ms

        if thumbnail_image is None:
            thumbnail_image = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB)

        df_f = df_total.loc[df_total.frame_id == frame_num]
        det_ids = list(df_f.agent_id)

        detections = []
        for det_id in det_ids:
            df_a = df_f.loc[df_f.agent_id == det_id]
            lat, lon, alt, fx, fy, fw, fh = float(df_a.latitude), float(df_a.longitude), float(df_a.altitude),\
                                            float(df_a.fx), float(df_a.fy), float(df_a.f_w), float(df_a.f_h)
            detections.append((frame_num, det_id, lat, lon, alt, fx, fy, fw, fh, ts))

        update_existing_and_create_new_tracklets_using_detection(tracklets, detections, fps)
        update_existing_tracklets_using_image_tracking(tracklets, frame_ori, frame_num, target_agent_id, trace,
                                                       ego_images_for_gif=ego_images_for_gif)

        buffer_size = 20
        trace_len = len(trace)
        if trace_len > buffer_size and (trace_len % buffer_size) == 0:
            trace_w_filtered = [trace[ii][2] for ii in range(trace_index, trace_len)]
            trace_h_filtered = [trace[ii][3] for ii in range(trace_index, trace_len)]

            from scipy.optimize import curve_fit
            # def objective(x, a, b, c, d, e, f):
            #     return a * (x ** 5) + b * (x ** 4) + c * (x ** 3) + d * (x ** 2) + e * x + f

            def objective(x, a, b):
                return a * x + b

            try:
                popt, _ = curve_fit(objective, trace_w_filtered, trace_h_filtered)
                # summarize the parameter values
                # a, b, c, d, e, f = popt
                a, b = popt

                w_min = np.min(trace_w_filtered)
                w_max = np.max(trace_w_filtered)
                num_points = trace_len - trace_index
                w_interval = (w_max - w_min) / num_points

                if trace_w_min < w_min < trace_w_max:
                    w_min = trace_w_max

                if trace_w_min < w_max < trace_w_max:
                    w_max = trace_w_min

                trace_w_filtered = []
                trace_h_filtered = []
                for jj in range(num_points):
                    w = w_min + jj * w_interval
                    # h = a * (w ** 5) + b * (w ** 4) + c * (w ** 3) + d * (w ** 2) + e * w
                    h = a * w + b

                    trace_w_filtered.append(w)
                    trace_h_filtered.append(h)

                if trace_len / buffer_size >= 3:
                    trace_w_filtered_total.extend(trace_w_filtered)
                    trace_h_filtered_total.extend(trace_h_filtered)

                trace_w_min = w_min
                trace_w_max = w_max
            except:
                pass

            trace_index = trace_len - 1

        if len(trace_w_filtered_total) >= 2:
            for i_trace in range(len(trace_w_filtered_total) - 1):
                i_trace_next = i_trace + 1
                fx, fy = trace_w_filtered_total[i_trace], trace_h_filtered_total[i_trace]
                fx_n, fy_n = trace_w_filtered_total[i_trace_next], trace_h_filtered_total[i_trace_next]
                x0, y0 = int(fx * input_video_width), int(fy * input_video_height)
                x1, y1 = int(fx_n * input_video_width), int(fy_n * input_video_height)
                cv2.line(frame_ori, (x0, y0), (x1, y1), (255, 0, 0), 12)

            if trace_len != prev_trace_len:
                img_gif = cv2.resize(cv2.cvtColor(frame_ori, cv2.COLOR_BGR2RGB), (480, 270))
                images_for_gif.append(img_gif)

        prev_trace_len = trace_len

        cv2.imshow(window_name, frame_ori)

        if auto:
            wait_time = 1
        else:
            wait_time = 0

        key = cv2.waitKey(wait_time) & 0xFF
        if key == escape_key or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(window_name)
            cap.release()
            sys.exit(0)
        elif key == ord("q"):
            break
        elif key == ord("n"):
            pass

        frame_num += 1

    if is_cam_gif:
        output_path = "."
        gif_name = os.path.basename(video_file)
        gif_name = gif_name.split(".")[0]
        imageio.mimsave(os.path.join(output_path, '%s.gif' % gif_name),
                        images_for_gif, duration=0.040)

    return ego_images_for_gif


def update_existing_tracklets_using_image_tracking(tracklets, frame, frame_num, target_agent_id, trace,
                                                   ego_images_for_gif=None):
    for tracklet in tracklets:
        if ego_images_for_gif is not None:
            frame_ori = frame.copy()

        tracking_status = tracklet.track(frame, frame_num)

        if tracking_status is False:
            tracklets.remove(tracklet)
        else:
            if tracklet.obj_id == target_agent_id:
                logging.info("testing vehicle: lat, lon %.8f, %.8f" % (tracklet.lat, tracklet.lon))
                xy_frac_pix = tracklet.center
                fx = xy_frac_pix[0]
                fy = xy_frac_pix[1]
                fw = np.abs((tracklet.offset_corner_2 - tracklet.offset_corner_3)[0])
                fh = np.abs((tracklet.offset_corner_1 - tracklet.offset_corner_2)[1])
                trace.append((tracklet.lat, tracklet.lon, fx, fy, fw, fh, tracklet.ts))
                if ego_images_for_gif is not None and frame_num % 25 == 0:
                    h, w = frame_ori.shape[0], frame_ori.shape[1]
                    w_start = int(np.clip(w * (fx - fw * 0.5), 0, w))
                    w_end = int(np.clip(w_start + w * fw, 0, w))
                    h_start = int(np.clip(h * (fy - fh * 0.5), 0, h))
                    h_end = int(np.clip(h_start + h * fh, 0, h))
                    frame_ego = frame_ori[h_start:h_end, w_start:w_end]
                    img_gif = cv2.resize(cv2.cvtColor(frame_ego, cv2.COLOR_BGR2RGB), (100, 100))
                    ego_images_for_gif.append(img_gif)


def update_existing_and_create_new_tracklets_using_detection(tracklets, detections, fps):
    global CUR_ID

    for _, proc_pre_dict in enumerate(detections):
        frame_number, index_det, lat, lon, alt, fx, fy, fw, fh, ts = proc_pre_dict

        center = np.array([fx, fy]).astype(float)
        half_vev_tr = np.array([fw * 0.5, -fh * 0.5]).astype(float)
        half_vev_br = np.array([fw * 0.5, fh * 0.5]).astype(float)
        half_vev_bl = np.array([-fw * 0.5, fh * 0.5]).astype(float)
        half_vev_tl = np.array([-fw * 0.5, -fh * 0.5]).astype(float)

        is_matched_existing_tracklet = False
        is_overlapped_existing_tracklet = False

        for index_obj in range(len(tracklets)):
            tracklet = tracklets[index_obj]

            is_to_update, is_overlap = tracklet.is_feature_update_qualified(center, half_vev_tr, half_vev_br,
                                                                            half_vev_bl, half_vev_tl, frame_number,
                                                                            index_det)

            is_overlapped_existing_tracklet |= is_overlap

            if is_to_update:
                tracklet.update_feature(center, half_vev_tr, half_vev_br, half_vev_bl, half_vev_tl, frame_number,
                                        lat, lon, ts)

                is_matched_existing_tracklet = True
                break

        is_new_obj = not (is_matched_existing_tracklet or is_overlapped_existing_tracklet)

        # Note: from now on only handle dynamic agents, registration zone
        is_in_registration_zone = True
        is_in_unsubscription_zone = False
        initial_time_window_for_reg = 1.0
        time_to_beginning = frame_number / fps

        is_to_register = is_new_obj \
                         and (is_in_registration_zone or
                              (time_to_beginning < initial_time_window_for_reg and not is_in_unsubscription_zone))

        if is_to_register:
            new_tracklet = Tracklet(CUR_ID, center, half_vev_tr, half_vev_br, half_vev_bl, half_vev_tl,
                                    frame_number, lat, lon, ts)

            if new_tracklet.is_initialized:
                CUR_ID += 1
                new_tracklet.register(index_det)
                tracklets.append(new_tracklet)


def fit_gauss(xdata, ydata, init=[1.0, 0.1, 0.2, 0.5]):
    """
    Note: parameterization for a Gaussian function: const, mu, sigma, offset
    :param xdata:
    :param ydata:
    :param init:
    :return:
    """
    fitfunc = lambda p, x: p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3]
    residuals = lambda a, x, y: y - fitfunc(a, x)

    par, cov, infodict, mesg, ier = optimize.leastsq(residuals, init, args=(xdata, ydata), full_output=True)

    ss_err = (infodict['fvec'] ** 2).sum()
    ss_tot = ((ydata - ydata.mean()) ** 2).sum()
    if ss_tot > 0:
        rsquared = 1 - (ss_err / ss_tot)
    else:
        rsquared = 0.

    if rsquared < 0.5:
        logging.info("fit_gauss: r squared %.3f" % rsquared)
        # TODO automatically tweak the fit parameters...
        # init = [1.0, 0.1, 0.01, 0.5]
        # return fit_gauss(xdata, ydata, init=init)

    return par, fitfunc(par, xdata)


def interpolate_2d_distribution(input_2d_dist, method):
    if method is None:
        return input_2d_dist

    x = np.arange(0, input_2d_dist.shape[1])
    y = np.arange(0, input_2d_dist.shape[0])
    array = np.ma.masked_invalid(input_2d_dist)
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    new_arr = array[~array.mask]

    output_interpolated = interpolate.griddata((x1, y1), new_arr.ravel(), (xx, yy), method=method)
    output_interpolated = np.clip(output_interpolated, 0, np.finfo(float).max)
    return output_interpolated

