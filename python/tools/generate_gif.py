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


def generate_gif(gps_file, gps_start, gps_steps, cam_file, cam_video, target_agent_id, auto, is_cam_gif, ts_start,
                 to_fit):
    gps_trace_df = load_gps(gps_file, gps_start, gps_steps)

    ego_imgs = load_cam_generate_gif(cam_file, cam_video, target_agent_id, auto, WINDOW_NAME, ESCAPE_KEY, WIDTH_WINDOW,
                          WINDOW_START_WIDTH, WINDOW_START_HEIGHT, ts_start, is_cam_gif)

    return gps_trace_df, ego_imgs


def main():
    parser = optparse.OptionParser()
    parser.add_option('--configs', action="store", default="")
    parser.add_option('--gps_file', action="store", default="")
    parser.add_option('--gps_start', action="store", default=1000, help=None)
    parser.add_option('--gps_steps', action="store", default=100, help=None)
    parser.add_option("--cam_file", action="store", default="")
    parser.add_option('--cam_video', action="store", default="")
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

    traces = []
    ego_gif_imgs = []

    if len(options.configs) > 0:
        configs = options.configs.split(",")
        n_config = 8
        n_cam = len(configs) // n_config
        for i in range(n_cam):
            to_fit, ts_start, a_id, g_start, g_steps, g_file, c_file, c_video =\
                configs[i * n_config: (i + 1) * n_config]

            to_fit, ts_start, a_id, g_start, g_steps, g_file, c_file, c_video =\
                eval(to_fit), int(ts_start), int(a_id), int(g_start), int(g_steps), g_file, c_file, c_video

            gps_trace, ego_imgs = generate_gif(g_file, g_start, g_steps, c_file, c_video, a_id, options.auto, options.cam_gif,
                                     ts_start, to_fit)

            traces.append(gps_trace)
            ego_gif_imgs.extend(ego_imgs)
    else:
        gps_trace, ego_imgs = generate_gif(options.gps_file, int(options.gps_start), int(options.gps_steps),
                                                  options.cam_file, options.cam_video, int(options.agent_id),
                                                  options.auto, options.cam_gif, int(options.ts_start), options.to_fit)
        traces.append(gps_trace)
        ego_gif_imgs.extend(ego_imgs)

    if options.map:
        df_total = pd.concat(traces)
        plot_map(df_total)

    if options.ego_gif:
        output_path = "."
        gif_name = "ego"
        imageio.mimsave(os.path.join(output_path, '%s.gif' % gif_name), ego_gif_imgs, duration=0.1)


if __name__ == "__main__":
    main()
