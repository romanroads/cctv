

set configs=^"True,-1,28,6800,800,C:\Users\element_symbolic_link\python\data\trip1.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4,^
True,-1,23,2200,600,C:\Users\element_symbolic_link\python\data\trip2.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4,^
True,1620368261800,5,3000,2200,C:\Users\element_symbolic_link\python\data\trip3.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4,^
False,-1,29,600,500,C:\Users\element_symbolic_link\python\data\trip4.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4,^
False,-1,8,2400,5000,C:\Users\element_symbolic_link\python\data\trip5.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.csv,C:\Users\element_symbolic_link\python\data\25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4^"

python tools\generate_gif.py --auto --configs %configs% --ego_gif

