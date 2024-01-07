

@echo off
set folder=C:\Users\element_symbolic_link\user_sample\python\data\
for %%x in (
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4
       ) do (
         echo %folder%%%x
         python demo.py --video_name %%x --video_path %folder% ^
        --model_path "%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_v0_model_rr_net_converted.pb" ^
        --calib_path "%folder%Calibration_Cams_25.yml" --auto --csv
         echo =-=-=-=-=-=
         echo.
       )

