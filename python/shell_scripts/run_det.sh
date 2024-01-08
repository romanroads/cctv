runs=("25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4" \
"25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4" \
"25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4" \
"25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4" \
"25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4")
for run in ${runs[*]};
do
echo "$run"
python demo.py --video_path "/home/element_cctv_symbolic_link/data/" \
--video_name "$run" \
--model_path /home/element_cctv_symbolic_link/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_v0_model_rr_net_converted.pb \
--calib_path /home/element_cctv_symbolic_link/data/Calibration_Cams_25.yml --csv --auto --output_video
done