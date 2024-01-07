
export data_path="/home/element_symbolic_link/user_sample/python/data/"

runs=("${data_path}25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4")

for run in ${runs[*]};
do
echo "$run"
python demo.py --video_path "$run" \
--model_path "${data_path}25_6a765b18-60bd-439c-a03f-295edd9d4b09_v0_model_rr_net_converted.pb" \
--calib_path "${data_path}Calibration_Cams_25.yml" --csv --auto
done