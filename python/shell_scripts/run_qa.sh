
DATA_PATH="/home/element_cctv_symbolic_link/data"

export configs=\
"True,-1,28,6800,800,${DATA_PATH}/trip1.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,"\
"True,-1,82,2200,600,${DATA_PATH}/trip2.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4,"\
"True,1620368261800,181,3000,2200,${DATA_PATH}/trip3.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4,"\
"False,-1,366,600,500,${DATA_PATH}/trip4.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4,"\
"False,-1,399,2400,5000,${DATA_PATH}/trip5.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4"

#export configs=\
#"True,-1,28,6800,800,${DATA_PATH}/trip1.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,"\
#"True,-1,82,2200,600,${DATA_PATH}/trip2.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,${DATA_PATH}/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4"

python tools/plot_rtk_gps_cam_det.py --auto --auto --configs "${configs}"
