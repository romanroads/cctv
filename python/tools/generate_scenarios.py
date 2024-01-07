import sys
import os
import optparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly_express as px
import cv2
import logging
import utm
import seaborn as sns
import imageio

sys.path.insert(0, ".")
from utility.compute_cte import compute_cte, compute_time_dependent_error
from utility.utils import compute_histogram_bins, load_gps, load_cam_generate_gif, compute_the_2d_weighted_histogram,\
    fit_gauss, interpolate_2d_distribution

plt.rcParams["font.family"] = "Times New Roman"

WINDOW_NAME = "RR RCIU QA Test"
WIDTH_WINDOW = 800
WINDOW_START_WIDTH = 50
WINDOW_START_HEIGHT = 50
ESCAPE_KEY = 27
X_0 = None
Y_0 = None


def plot_map(df_total):
    fig = px.scatter_mapbox(df_total, lat="Latitude", lon="Longitude",
                            color_continuous_scale=px.colors.sequential.Rainbow, size_max=40,
                            zoom=18, height=1000, mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def generate_scenarios(video_path, video_name, auto, is_cam_gif, ts_start,
                 to_fit):

    # TODO for now just support mp4 offline files and rtsp steaming data
    data_type = None
    if ".mp4" in video_name:
        cv_source_name = os.path.join(video_path, video_name)
        data_type = "mp4"
        if not os.path.exists(cv_source_name):
            logging.warning("generate_scenarios: video source file %s does not exist!" % cv_source_name)
            sys.exit(0)
        det_csv_file_name = cv_source_name.replace(".mp4", ".csv")

    elif "rtsp" in video_name:
        data_type = "rtsp"
        cv_source_name = video_name
        sys.exit(0)
    else:
        logging.warning("generate_scenarios: video name %s not supported!" % video_name)
        sys.exit(0)

    # TODO no ego car is defined in scenario extraction
    target_agent_id = -1
    if not os.path.exists(cv_source_name):
        return []

    return load_cam_generate_gif(det_csv_file_name, cv_source_name, target_agent_id, auto, WINDOW_NAME, ESCAPE_KEY,
                                 WIDTH_WINDOW, WINDOW_START_WIDTH, WINDOW_START_HEIGHT, ts_start, is_cam_gif)


def main():
    parser = optparse.OptionParser()
    parser.add_option('--configs', action="store", default="")
    parser.add_option('--gps_file', action="store", default="")
    parser.add_option('--gps_start', action="store", default=1000, help=None)
    parser.add_option('--gps_steps', action="store", default=100, help=None)
    parser.add_option("--video_path", action="store", default="")
    parser.add_option('--video_name', action="store", default="")
    parser.add_option('--logging', action="store", default="INFO", help="logging level")
    parser.add_option('--auto', action="store_true", default=False, help="automatically run pipeline")
    parser.add_option('--agent_id', action="store", default=-1, help="select a target agent")
    parser.add_option('--map', action="store_true", default=False)
    parser.add_option('--ts_start', action="store", default=-1, help=None)
    parser.add_option('--to_fit', action="store_true", default=False, help=None)
    parser.add_option('--cam_gif', action="store_true", default=False, help=None)
    parser.add_option('--ego_gif', action="store_true", default=False, help=None)
    options, args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=options.logging.upper())

    ego_imgs = generate_scenarios(options.video_path, options.video_name, options.auto, options.cam_gif,
                                  int(options.ts_start), options.to_fit)

    if options.map:
        df_total = pd.concat(traces)
        plot_map(df_total)

    if options.ego_gif:
        output_path = "."
        gif_name = "ego"
        imageio.mimsave(os.path.join(output_path, '%s.gif' % gif_name), ego_imgs, duration=0.1)


if __name__ == "__main__":
    main()
