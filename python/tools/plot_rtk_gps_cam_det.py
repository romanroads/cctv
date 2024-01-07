import sys
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

sys.path.insert(0, ".")
from utility.compute_cte import compute_cte, compute_time_dependent_error
from utility.utils import compute_histogram_bins, load_gps, load_cam, compute_the_2d_weighted_histogram, fit_gauss,\
    interpolate_2d_distribution

plt.rcParams["font.family"] = "Times New Roman"

WINDOW_NAME = "RR RCIU QA Test"
WIDTH_WINDOW = 800
WINDOW_START_WIDTH = 50
WINDOW_START_HEIGHT = 50
ESCAPE_KEY = 27
X_0 = None
Y_0 = None


def compute_statistics(gps_file, gps_start, gps_steps, cam_file, cam_video, target_agent_id, auto, is_plot_map,
                       ts_start, to_fit):
    list_traces = []

    gps_trace_df = load_gps(gps_file, gps_start, gps_steps)
    list_traces.append(gps_trace_df)

    target_trace_df, thumbnail_image = load_cam(cam_file, cam_video, target_agent_id, auto, WINDOW_NAME, ESCAPE_KEY,
                                                WIDTH_WINDOW, WINDOW_START_WIDTH, WINDOW_START_HEIGHT, ts_start)

    list_traces.append(target_trace_df)

    df_total = pd.concat(list_traces)

    if is_plot_map:
        plot_map(df_total)

    # Note: A, B, C and D 4 different metrics, A is CTE which does not need timestamp
    tuple_stat = compute_metrics(target_trace_df, gps_trace_df, to_fit)
    return tuple_stat, thumbnail_image


def compute_metrics(df_meas, df_gt, to_fit):
    global X_0, Y_0

    column_names = ['Latitude', 'Longitude', 'Timestamp', 'Fx', 'Fy', 'Fw', 'Fh']
    meas = df_meas[column_names].to_numpy()
    gt = df_gt[column_names].to_numpy()

    meas_space = []
    meas_time = []
    meas_image_loc = []
    for i in range(meas.shape[0]):
        lat, lon, ts, fx, fy, fw, fh = meas[i]
        utm_coord = utm.from_latlon(lat, lon)
        x, y = utm_coord[0], utm_coord[1]
        if i == 0 and X_0 is None and Y_0 is None:
            X_0, Y_0 = x, y
        x -= X_0
        y -= Y_0
        meas_space.append((x, y))
        meas_time.append(ts)
        meas_image_loc.append((fx, fy, fw, fh))

    gt_space = []
    gt_time = []
    gt_image_loc = []
    for i in range(gt.shape[0]):
        lat, lon, ts, fx, fy, fw, fh = gt[i]
        utm_coord = utm.from_latlon(lat, lon)
        x, y = utm_coord[0], utm_coord[1]
        x -= X_0
        y -= Y_0
        gt_space.append((x, y))
        gt_time.append(ts)
        gt_image_loc.append((fx, fy, fw, fh))

    meas_space = np.array(meas_space)
    gt_space = np.array(gt_space)
    meas_time = np.array(meas_time)
    gt_time = np.array(gt_time)

    # Note: A, B, C and D 4 different metrics, A is CTE which does not need timestamp
    # B, C and D needs timestamp to make measurements at each timestamp
    metric_a_tuple, x_fit, y_fit, x_meas, y_meas = compute_cte(meas_space, gt_space, meas_image_loc, to_fit=to_fit)
    metric_b_tuple, metric_c_tuple, metric_d_tuple = compute_time_dependent_error(meas_space, meas_time, gt_space,
                                                                               gt_time, meas_image_loc)
    return metric_a_tuple, metric_b_tuple, metric_c_tuple, metric_d_tuple, x_fit, y_fit, x_meas, y_meas


def plot_map(df_total):
    fig = px.scatter_mapbox(df_total, lat="Latitude", lon="Longitude",
                            color_continuous_scale=px.colors.sequential.Rainbow, size_max=40,
                            zoom=18, height=1000, mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_metrics(title, index_obs, img, stats, thumbnail_image, color_codes, color_code, x_start, x_end, x_bins,
                 y_start):
    f = plt.figure(figsize=(18, 6))
    ax1 = f.add_subplot(121)
    ax1.set_xlabel(title, fontsize=30)
    ax1.set_ylabel("Count", fontsize=30)

    ax1.set_xlim(x_start, x_end)

    x_bin_center, x_bin_edge = compute_histogram_bins(x_bins, x_start, x_end)

    total_obs = []
    h, w, c = thumbnail_image.shape
    h_heat = 100
    w_heat = 100
    h_edges, w_edges = compute_the_2d_weighted_histogram(0, h, h_heat, 0, w, w_heat)
    cte_heatmap = np.zeros((h_heat, w_heat))
    cte_heatmap_counts = np.zeros((h_heat, w_heat))

    n_trace = len(stats)
    n_trace_with_obs = 0
    for i in range(n_trace):
        tuple_stat = stats[i]
        obs_tuple = tuple_stat[index_obs]

        obs = [obs_tuple[ii][0] for ii in range(len(obs_tuple))]
        fx = [obs_tuple[ii][1][0] for ii in range(len(obs_tuple))]
        fy = [obs_tuple[ii][1][1] for ii in range(len(obs_tuple))]
        fw = [obs_tuple[ii][1][2] for ii in range(len(obs_tuple))]
        fh = [obs_tuple[ii][1][3] for ii in range(len(obs_tuple))]

        fx, fy, fw, fh = np.array(fx), np.array(fy), np.array(fw), np.array(fh)
        fx, fy, fw, fh = fx * w, fy * h, fw * w, fh * h
        fx, fy, fw, fh = fx.astype(int), fy.astype(int), fw.astype(int), fh.astype(int)

        h_counts, _, _ = np.histogram2d(fy, fx, bins=(h_edges, w_edges))
        h_weights, _, _ = np.histogram2d(fy, fx, bins=(h_edges, w_edges), weights=np.abs(obs))
        h_weighted_ave = h_weights / h_counts
        h_weighted_ave = np.nan_to_num(h_weighted_ave, nan=0)
        cte_heatmap += h_weighted_ave
        cte_heatmap_counts += (h_counts > 0).astype(int)

        total_obs.extend(obs)
        color_trace = color_codes[i]
        n_obs = len(obs)
        if n_obs > 0:
            n, bins, patches = plt.hist(obs, bins=x_bin_edge, color=color_trace, histtype="step", lw=3,
                                        label="Trip%s" % (i + 1))
            n_trace_with_obs += 1

    cte_heatmap = cte_heatmap / cte_heatmap_counts
    cte_heatmap = interpolate_2d_distribution(cte_heatmap, None)

    if n_trace_with_obs > 1:
        n, bins, patches = plt.hist(total_obs, bins=x_bin_edge, color=color_code, histtype="step", lw=3,
                                    label="Total")

    if n_trace_with_obs > 0:
        y_end = np.max(n) * 1.2
        ax1.set_ylim(y_start, y_end)

        x_width_logo = (x_end - x_start) * 0.3
        y_height_logo = (y_end - y_start) * 0.25
        x_start_logo = x_start + (x_end - x_start) * 0.05
        y_start_logo = y_start + (y_end - y_start) * 0.65
        ax1.imshow(img, extent=[x_start_logo, x_start_logo + x_width_logo, y_start_logo, y_start_logo + y_height_logo],
                   aspect="auto")

        c, y_fit = fit_gauss(x_bin_center, n)
        fit_result_string = "Fit Parameters:\n$\mu$: %.3f[m]\n$\sigma$: %.3f [m]\n" % (c[1], abs(c[2]))
        ax1.text(x_start + (x_end - x_start) * 0.6, y_start + (y_end - y_start) * 0.4, fit_result_string, size=28)
        ax1.plot(x_bin_center, y_fit, color="orange", lw=4)
        ax1.legend(loc="upper right")

        ax2 = f.add_subplot(122)
        cmap = plt.cm.get_cmap("plasma")
        heatmap = sns.heatmap(cte_heatmap, cmap=cmap, alpha=0.5, zorder=2)
        heatmap.imshow(thumbnail_image, aspect=heatmap.get_aspect(), extent=heatmap.get_xlim() + heatmap.get_ylim(),
                       zorder=1)

    plt.tight_layout()


def plot_trace(stats, color_codes, img):
    n_trace = len(stats)
    f = plt.figure(figsize=(6, 6))
    ax1 = f.add_subplot(111)
    ax1.set_xlabel("X [m]", fontsize=30)
    ax1.set_ylabel("Y [m]", fontsize=30)
    x_start, x_end = np.finfo(float).max, np.finfo(float).min
    y_start, y_end = np.finfo(float).max, np.finfo(float).min
    for i in range(n_trace):
        c = color_codes[i]
        tuple_stat = stats[i]
        x_fit, y_fit, x_meas, y_meas = tuple_stat[-4:]
        ax1.plot(x_fit, y_fit, '--', label="Trip%s (GPS)" % (i + 1), color=c)
        ax1.scatter(x_meas, y_meas, label="Trip%s (RCIU)" % (i + 1), color=c)
        x_end = np.max(x_fit) if np.max(x_fit) > x_end else x_end
        x_start = np.min(x_fit) if np.min(x_fit) < x_start else x_start
        y_end = np.max(y_fit) if np.max(y_fit) > y_end else y_end
        y_start = np.min(y_fit) if np.min(y_fit) < y_start else y_start
    ax1.legend(loc="upper right")

    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)

    x_width_logo = (x_end - x_start) * 0.3
    y_height_logo = (y_end - y_start) * 0.2
    x_start_logo = x_start + (x_end - x_start) * 0.05
    y_start_logo = y_start + (y_end - y_start) * 0.05
    ax1.imshow(img, extent=[x_start_logo, x_start_logo + x_width_logo, y_start_logo, y_start_logo + y_height_logo],
               aspect="auto")

    plt.tight_layout()


def plot_delta_d_vs_dist(stats, img):
    index_obs = 0
    n_trace = len(stats)

    dist_start = 0.
    dist_end = 120.
    dist_bin = 10
    dist_interval = (dist_end - dist_start) / dist_bin
    list_of_list_obs = []
    for i in range(dist_bin + 1):
        list_of_list_obs.append([])

    for i in range(n_trace):
        tuple_stat = stats[i]
        obs_tuple = tuple_stat[index_obs]

        obs = [obs_tuple[ii][0] for ii in range(len(obs_tuple))]

        x_cam, y_cam = -5, 20
        dists = [np.sqrt((obs_tuple[ii][2][0] - x_cam) ** 2 +
                         (obs_tuple[ii][2][1] - y_cam) ** 2) for ii in range(len(obs_tuple))]

        for j in range(len(dists)):
            d = dists[j]
            d_index = int((d - dist_start) / dist_interval)
            if d_index < 0 or d_index > dist_bin:
                continue
            list_of_list_obs[d_index].append(obs[j])

    d_s = []
    obs_s = []
    obs_error_s = []
    mu_0 = None
    for i in range(dist_bin + 1):
        list_obs = list_of_list_obs[i]
        if len(list_obs) <= 0:
            continue

        hist = np.array(list_obs)
        mu = np.mean(hist)

        if mu_0 is None:
            # Note use 0.119 from our final opt results shown in QA report
            mu_0 = mu - 0.119

        sigma = np.sqrt(np.var(hist))
        d_s.append(i * dist_interval + dist_start)
        obs_s.append(mu - mu_0)
        obs_error_s.append(sigma)

    f = plt.figure(figsize=(6, 6))
    ax1 = f.add_subplot(111)
    ax1.set_xlabel("Dist to Sensor [m]", fontsize=30)
    ax1.set_ylabel("Uncertainty [m]", fontsize=30)
    x_start, x_end = dist_start - 1, dist_end + 1
    y_start, y_end = -0.01, np.max(obs_s) * 1.2

    ax1.errorbar(d_s, obs_s, xerr=None, yerr=obs_error_s, fmt='-o')

    ax1.set_xlim(x_start, x_end)
    ax1.set_ylim(y_start, y_end)

    x_width_logo = (x_end - x_start) * 0.3
    y_height_logo = (y_end - y_start) * 0.2
    x_start_logo = x_start + (x_end - x_start) * 0.05
    y_start_logo = y_start + (y_end - y_start) * 0.75
    ax1.imshow(img, extent=[x_start_logo, x_start_logo + x_width_logo, y_start_logo, y_start_logo + y_height_logo],
               aspect="auto")

    plt.tight_layout()


def plot_statistics(stats, thumbnail_image):
    img = Image.open("./artifacts/ROMAN_ROADS_LOGO_COLOR.png")
    img.thumbnail((500, 500), Image.ANTIALIAS)

    color_codes = [np.random.rand(3, ) for _ in range(len(stats))]
    color_code = np.random.rand(3, )

    plot_metrics("Cross-track-error [m]", 0, img, stats, thumbnail_image, color_codes, color_code,
                 -2, 2, 16, 0)
    plot_metrics("Delta D [m]", 1, img, stats, thumbnail_image, color_codes, color_code,
                 -3, 3, 16, 0)
    plot_metrics("Delta D Lateral [m]", 2, img, stats, thumbnail_image, color_codes, color_code,
                 -3, 3, 31, 0)
    plot_metrics("Delta D Longitudinal [m]", 3, img, stats, thumbnail_image, color_codes, color_code,
                 -3, 3, 31, 0)

    plot_trace(stats, color_codes, img)

    plot_delta_d_vs_dist(stats, img)

    plt.show()


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
    options, args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=options.logging.upper())

    if len(options.configs) > 0:
        configs = options.configs.split(",")
        n_config = 8
        n_cam = len(configs) // n_config
        stats = []
        for i in range(n_cam):
            to_fit, ts_start, a_id, g_start, g_steps, g_file, c_file, c_video =\
                configs[i * n_config: (i + 1) * n_config]

            to_fit, ts_start, a_id, g_start, g_steps, g_file, c_file, c_video =\
                eval(to_fit), int(ts_start), int(a_id), int(g_start), int(g_steps), g_file, c_file, c_video

            tuple_stat, thumbnail_image = compute_statistics(g_file, g_start, g_steps, c_file, c_video, a_id, options.auto,
                                                      options.map, ts_start, to_fit)
            stats.append(tuple_stat)

        plot_statistics(stats, thumbnail_image)
    else:
        tuple_stat, thumbnail_image = compute_statistics(options.gps_file, int(options.gps_start), int(options.gps_steps),
                                                  options.cam_file, options.cam_video, int(options.agent_id),
                                                  options.auto, options.map, int(options.ts_start), options.to_fit)
        plot_statistics([tuple_stat], thumbnail_image)


if __name__ == "__main__":
    main()
