# Copyright 2019 - 2024 The ROMAN ROADS Developers. All Rights Reserved.
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
import optparse
import timeit
import logging
import numpy as np
import cv2
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pandas as pd

import tensorflow as tf
import tensorflow.compat.v1 as tfc
from tensorflow.compat.v1 import ConfigProto

from utility.load_graph import load_graph
from utility.load_decryption_key import load_decryption_key
from utility.load_cam_par import load_cam_par
from utility.compute_transformation_matrix import compute_transformation_matrix, compute_estimate_local_z_component

from constants import *


def demo(video_path, video_name, model_path, cam_par_path, auto, visualize_mask, dump_csv, frame_max,
         output_video, no_gui=False):
    decryption_key = load_decryption_key()

    if frame_max < 0:
        frame_max = np.iinfo(np.int32).max

    k_inv, t_vector, r_transpose, matrix_geo, c, ax, ay, b = load_cam_par(cam_par_path)

    matrix_a, vector_b = compute_transformation_matrix(k_inv, t_vector, r_transpose, matrix_geo)

    # TODO for now just support mp4 offline files and rtsp steaming data
    data_type = None
    if ".mp4" in video_name:
        cv_source_name = os.path.join(video_path, video_name)
        data_type = "mp4"
        if not os.path.exists(cv_source_name):
            logging.warning("demo: video source file %s does not exist!" % cv_source_name)
            sys.exit(0)
    elif "rtsp" in video_name:
        data_type = "rtsp"
        cv_source_name = video_name
    else:
        logging.warning("demo: video name %s not supported!" % video_name)
        sys.exit(0)

    cap = cv2.VideoCapture(cv_source_name)
    input_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info("demo: input video has %s frames, with frame size w (%s) x h (%s) and fps %.1f, frame_max = %s" %
                 (frame_count, input_video_width, input_video_height, fps, frame_max))

    if output_video:
        video_file_name_output = os.path.join(video_path, "%s_output.mp4" % (video_name.split(".")[0]))
        outvideo_writer = cv2.VideoWriter(video_file_name_output, cv2.VideoWriter_fourcc(*'mp4v'),
                                          fps, (input_video_width, input_video_height))
        logging.info("output video at %s" % video_file_name_output)

    if data_type == "mp4":
        if frame_count <= 0:
            return
    elif data_type == "rtsp":
        frame_count = np.iinfo(np.int32).max

    if not no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_EXPANDED)

    height_window = int(WIDTH_WINDOW * (input_video_height * 1. / input_video_width))

    if not no_gui:
        cv2.resizeWindow(WINDOW_NAME, WIDTH_WINDOW, height_window)
        cv2.moveWindow(WINDOW_NAME, WINDOW_START_WIDTH, WINDOW_START_HEIGHT)

    # Note: cache the computed traces
    cached_trace_data = []

    with tf.device("/GPU:0"):
        graph = load_graph(model_path, is_decryption_needed=True, decryption_key=decryption_key)

        try:
            hybrid_box_fixed_len = graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)
        except KeyError:
            logging.error("demo: the DNN model file is corrupted!")
            return

        config = ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        config.gpu_options.visible_device_list = '0'
        config.allow_soft_placement = False

        with tfc.Session(graph=graph, config=config) as sess:
            frame_num = 0
            while cap.isOpened() and frame_num < frame_count and frame_num < frame_max:
                start_time = timeit.default_timer()

                _ret, frame_ori = cap.read()

                if not _ret:
                    break

                if data_type == "mp4":
                    desired_height, desired_width, _ = frame_ori.shape
                    black_mask_for_objects = np.zeros((desired_height, desired_width), dtype=np.uint8)
                    frame_for_objects = np.full((desired_height, desired_width, 3), BLUE_BRG, dtype=np.uint8)

                    frame_dnn = cv2.resize(frame_ori, (IMAGE_WIDTH_ENDTER_DNN, IMAGE_HEIGHT_ENDTER_DNN))
                    frame_dnn = cv2.cvtColor(frame_dnn, cv2.COLOR_BGR2RGB)

                    input_data = frame_dnn.flatten()
                    hybrid_box_fixed_len_arr \
                        = sess.run([hybrid_box_fixed_len],
                                   feed_dict={
                                       "input:0": input_data  # x : frame should be used if using the original input,
                                       # but we assume we always convert original input layer to our non-shaped input for ML
                                       # net that has bugs for shape input tensor, has to be [None,]
                                   })

                    hybrid_box_fixed_len_arr = hybrid_box_fixed_len_arr[0]
                    num_float_in_output_tensor = len(hybrid_box_fixed_len_arr)
                    num_float_per_agent = int(num_float_in_output_tensor / NUM_FIXED_AGENTS)
                    hybrid_box_fixed_len_arr = np.reshape(hybrid_box_fixed_len_arr, (NUM_FIXED_AGENTS, num_float_per_agent))

                    for i in range(NUM_FIXED_AGENTS):
                        x0, y0, x1, y1, score_i, label_i = hybrid_box_fixed_len_arr[i][0:6]

                        # Note these are padded value to make the output fixed length
                        if score_i <= -1:
                            break

                        if score_i < THRESHOLD_CONFIDENCE_SCORE:
                            continue

                        f_x0 = np.clip(x0 * 1. / IMAGE_WIDTH_ENDTER_DNN, 0, 1)
                        f_x1 = np.clip(x1 * 1. / IMAGE_WIDTH_ENDTER_DNN, 0, 1)
                        f_y0 = np.clip(y0 * 1. / IMAGE_HEIGHT_ENDTER_DNN, 0, 1)
                        f_y1 = np.clip(y1 * 1. / IMAGE_HEIGHT_ENDTER_DNN, 0, 1)
                        f_x_center = (f_x0 + f_x1) * 0.5
                        f_y_center = (f_y0 + f_y1) * 0.5
                        fractional_range_x = np.abs(f_x0 - f_x1)
                        fractional_range_y = np.abs(f_y0 - f_y1)
                        f_y_center_to_map = f_y_center + fractional_range_y * 0.25

                        # Note from now on x and y are in original frame size
                        x0, y0, x1, y1 = int(f_x0 * desired_width), int(f_y0 * desired_height), \
                                         int(f_x1 * desired_width), int(f_y1 * desired_height)

                        # Note: projection matrices used to transform from 2D image to 3D world
                        dist = compute_estimate_local_z_component(f_x_center, f_y_center_to_map, c, ax, ay, b)
                        uv_4x1 = np.array([f_x_center * dist, f_y_center_to_map * dist, dist, 1.]).reshape(4, 1).astype(float)
                        pos_world_other = np.matmul(matrix_a, uv_4x1) - vector_b

                        lon, alt, lat = pos_world_other[0][0], pos_world_other[1][0], pos_world_other[2][0]

                        cached_trace_data.append((frame_num, i, lat, lon, alt, f_x_center, f_y_center, fractional_range_x,
                                                  fractional_range_y))

                        if visualize_mask:
                            list_of_mask_points_i = []

                            mask_i_empty = np.zeros((desired_height, desired_width), bool)

                            # Note: this is the low resolution instance semantic map
                            mask_i = hybrid_box_fixed_len_arr[i][6:6 + NUM_MASK_POINTS * NUM_MASK_POINTS]. \
                                reshape((NUM_MASK_POINTS, NUM_MASK_POINTS))

                            binary_mask_i = mask_i > 0.5

                            seg_map = SegmentationMapsOnImage(binary_mask_i, shape=binary_mask_i.shape)

                            x_range = int(round(x1 - x0))
                            y_range = int(round(y1 - y0))

                            seg_map = seg_map.resize((y_range, x_range))
                            scaled_mask = seg_map.get_arr()

                            mask_i_empty[y0:y0 + y_range, x0:x0 + x_range] = scaled_mask

                            mask_indices = np.argwhere(mask_i_empty == True).astype(float)
                            mask_indices = mask_indices.astype(int)

                            for point in mask_indices:
                                list_of_mask_points_i.append((point[1], point[0]))

                            # Note: for visualization purposes
                            poly_mask_cords = np.array(list_of_mask_points_i, dtype=np.int32)
                            black_bkg_mask_fg_i = np.zeros((desired_height, desired_width), dtype=np.uint8)
                            cv2.fillPoly(black_bkg_mask_fg_i, [poly_mask_cords], WHITE)
                            black_mask_for_objects = black_bkg_mask_fg_i | black_mask_for_objects

                        cv2.rectangle(frame_ori, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        # Note: end of the for loop over all agents on this frame

                    if visualize_mask:
                        frame_for_objects = cv2.bitwise_and(frame_for_objects, frame_for_objects,
                                                            mask=black_mask_for_objects)
                        alpha = 0.7
                        frame_ori = cv2.addWeighted(frame_for_objects, alpha, frame_ori, 1, 0)

                if not no_gui:
                    cv2.imshow(WINDOW_NAME, frame_ori)

                if output_video:
                    outvideo_writer.write(frame_ori)

                computation_time = timeit.default_timer() - start_time
                computation_fps = 1 / computation_time
                logging.info("frame number %s, computation fps %.1f" % (frame_num, computation_fps))

                if auto:
                    wait_time = 1
                else:
                    wait_time = 0

                key = cv2.waitKey(wait_time) & 0xFF

                if not no_gui:
                    if key == ESCAPE_KEY or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        cv2.destroyWindow(WINDOW_NAME)
                        cap.release()
                        sys.exit(0)
                    elif key == ord("q"):
                        break
                    elif key == ord("n"):
                        pass
                else:
                    pass

                frame_num += 1

    if dump_csv and len(cached_trace_data) > 0:
        column_names = ["frame_id", "agent_id", "latitude", "longitude", "altitude", "fx", "fy", "f_w", "f_h"]
        df_total = pd.DataFrame(cached_trace_data, columns=column_names)
        if data_type == "mp4":
            file_name = os.path.basename(video_name).split(".")[0]
        elif data_type == "rtsp":
            file_name = os.path.basename(video_name).split("@")[1].split("/")[0]
        else:
            file_name = "temp"

        file_folder = os.path.join(REPO_PATH, 'data')
        os.makedirs(file_folder, exist_ok=True)
        csv_path = os.path.join(file_folder, "%s.csv" % file_name)
        df_total.to_csv(csv_path, float_format="%.8f", index=False)

    logging.info("processed %s frames in total" % frame_num)
    cap.release()

    if output_video:
        outvideo_writer.release()


def main():
    parser = optparse.OptionParser()
    parser.add_option('--video_path', action="store", default=r"./")
    parser.add_option('--video_name', action="store", default=r"your_video.mp4")
    parser.add_option('--model_path', action="store", default=r"your_dnn_model.pb")
    parser.add_option('--calib_path', action="store", default=r"your_calib.yml")
    parser.add_option('--logging', action="store", default="INFO", help=None)
    parser.add_option('--auto', action="store_true", default=False, help=None)
    parser.add_option('--vis_mask', action="store_true", default=False, help=None)
    parser.add_option('--csv', action="store_true", default=False, help=None)
    parser.add_option('--frame_max', action="store", default=-1, help=None)
    parser.add_option('--output_video', action="store_true", default=False, help=None)

    options, args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=options.logging.upper())

    demo(options.video_path, options.video_name, options.model_path, options.calib_path, options.auto, options.vis_mask,
         options.csv, int(options.frame_max), options.output_video, no_gui=True)


if __name__ == "__main__":
    main()
