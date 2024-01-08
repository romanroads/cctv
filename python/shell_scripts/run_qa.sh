
#export configs=\
#"28,6800,800,/home/joe/Downloads/trip1.csv,/home/joe/work/element/python/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,/home/joe/Downloads/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,"\
#"82,2200,600,/home/joe/Downloads/trip2.csv,/home/joe/work/element/python/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,/home/joe/Downloads/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4,"\
#"181,3000,2200,/home/joe/Downloads/trip3.csv,/home/joe/work/element/python/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.csv,/home/joe/Downloads/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4,"\
#"366,600,500,/home/joe/Downloads/trip4.csv,/home/joe/work/element/python/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.csv,/home/joe/Downloads/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4,"\
#"399,2400,5000,/home/joe/Downloads/trip5.csv,/home/joe/work/element/python/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.csv,/home/joe/Downloads/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4"

export configs=\
"28,6800,800,/home/element_cctv_symbolic_link/data/trip1.csv,/home/element_cctv_symbolic_link/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,/home/element_cctv_symbolic_link/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,"\
"82,2200,600,/home/element_cctv_symbolic_link/data/trip2.csv,/home/element_cctv_symbolic_link/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,/home/element_cctv_symbolic_link/data/25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4"

python tools/plot_rtk_gps_cam_det.py --auto --auto --configs "$configs"
