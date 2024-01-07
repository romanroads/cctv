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
from shapely.geometry import Polygon
import cv2

IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP = 0.1


class Tracklet:
    def __init__(self, obj_id, center, half_vev_tr, half_vev_br, half_vev_bl, half_vev_tl, frame_counter, lat, lon, ts):
        self.obj_id = obj_id
        self.frame_counter = frame_counter
        self.last_detected_frame_id = frame_counter

        self.lat = lat
        self.lon = lon
        self.ts = ts

        self.center = center
        self.prev_center = center
        self.traj = [center]
        self.offset_corner_1 = half_vev_tr
        self.offset_corner_2 = half_vev_br
        self.offset_corner_3 = half_vev_bl
        self.offset_corner_4 = half_vev_tl

        self.bounding_polygon = None
        self.top_left_corner = None
        self.bottom_right_corner = None
        self.update_bounding_polygon()

        self.position_updated_by_detection = True
        self.is_initialized = True

    def update_feature(self, center, half_vev_tr, half_vev_br, half_vev_bl, half_vev_tl, frame_counter, lat, lon, ts):
        self.set_last_seen_frame_id(frame_counter)

        self.center = center

        self.offset_corner_1 = half_vev_tr
        self.offset_corner_2 = half_vev_br
        self.offset_corner_3 = half_vev_bl
        self.offset_corner_4 = half_vev_tl

        self.traj.append(center)
        self.prev_center = center
        self.update_bounding_polygon()

        self.lat = lat
        self.lon = lon
        self.ts = ts

        self.position_updated_by_detection = True

    def track(self, frame, frame_id):
        h, w = frame.shape[0], frame.shape[1]
        x0, y0 = self.top_left_corner[0] * w, self.top_left_corner[1] * h
        x1, y1 = self.bottom_right_corner[0] * w, self.bottom_right_corner[1] * h
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        x_corner_1, y_corner_1 = x0, y0
        cv2.putText(frame, "agent_%s" % self.obj_id, (x_corner_1, y_corner_1), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        staled_frames = frame_id - self.last_detected_frame_id
        status = True
        if staled_frames > 10:
            status = False
        return status

    def update_bounding_polygon(self):
        self.bounding_polygon = Polygon([self.prev_center + self.offset_corner_1,
                                         self.prev_center + self.offset_corner_2,
                                         self.prev_center + self.offset_corner_3,
                                         self.prev_center + self.offset_corner_4])
        self.top_left_corner = self.prev_center + self.offset_corner_4
        self.bottom_right_corner = self.prev_center + self.offset_corner_2

    def set_last_seen_frame_id(self, f_id):
        self.last_detected_frame_id = f_id

    def is_feature_update_qualified(self, center, half_vev_tr, half_vev_br, half_vev_bl, half_vev_tl,
                                    frame_number, index_det=None):

        bounding_polygon = Polygon([center + half_vev_tr,
                                    center + half_vev_br,
                                    center + half_vev_bl,
                                    center + half_vev_tl])

        is_overlap = self.is_duplicate(bounding_polygon, index_det)

        # Note: if not overlapped, we do not need to consider whether or not to update existing tracklet
        if not is_overlap:
            return False, is_overlap

        # Note: even it is overlapped physically, we need to look at historical data of vehicle length, width to
        # determine whether or not this is a real match for update
        is_matching = True

        self.set_last_seen_frame_id(frame_number)

        return is_matching, is_overlap

    def is_duplicate(self, bounding_polygon, index_detection=None):
        intersection = self.bounding_polygon.intersection(bounding_polygon).area
        union = self.bounding_polygon.union(bounding_polygon).area
        iou = intersection / union
        is_iou_too_large = iou > IOU_BETWEEN_TWO_POLYGON_TO_AVOID_DUP

        return is_iou_too_large

    def unregister(self, note):
       pass

    def register(self, index_detection=None, index_pose_tracker=None):
        pass
