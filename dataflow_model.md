## This file Describles the current working model of the extracion 

First Yolopose generate the posture data and the geometry constructions of the files ->pose.yaml
Second Perspective3P reads the posture data and pick out the points with geometry constraction then use perspective 3p to solve out the Tvecs and Rvecs thus establish the camera parameters and best made out the floor situation and extristic parameters of the camera which enables us to a point to point projection of floor plane to the img plane. 

The major goal of this project is to estimate the position of the teacher with only one point which at minial cost.