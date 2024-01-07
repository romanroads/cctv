
set folder=C:\Users\element_symbolic_link\user_sample\python\data\

set configs=^"True,-1,28,6800,800,%folder%trip1.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,^
True,-1,23,2200,600,%folder%trip2.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4,^
True,1620368261800,5,3000,2200,%folder%trip3.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4,^
False,-1,29,600,500,%folder%trip4.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4,^
False,-1,8,2400,5000,%folder%trip5.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.csv,%folder%25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4^"

python tools\plot_rtk_gps_cam_det.py --auto --configs %configs%
