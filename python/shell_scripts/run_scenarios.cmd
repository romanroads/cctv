
@echo off
set folder=C:\Users\element_symbolic_link\user_sample\python\data\
set data_name=25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4

python demo.py --video_name %data_name% ^
--model_path %folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_v0_model_rr_net_converted.pb ^
--calib_path %folder%Calibration_Cams_25.yml --auto --csv --video_path %folder%

echo =-=-=-=-=-=

python tools\generate_scenarios.py --auto --video_path %folder% --video_name %data_name%
