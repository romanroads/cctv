
@echo off
set folder=C:\Users\joexi\AppData\Local\Temp\RomanRoads\element\
for %%x in (
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run0_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run3_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run6_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run9_Trip0_User.mp4
        25_6a765b18-60bd-439c-a03f-295edd9d4b09_Exp4_Run12_Trip0_User.mp4
       ) do (
         echo %folder%%%x
         echo s3://user-upload-data/%%x
         aws --profile cn s3 cp s3://user-upload-data/%%x %folder%%%x
         echo =-=-=-=-=-=
         echo.
       )
