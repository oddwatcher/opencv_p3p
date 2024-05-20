import cv2 as cv
import numpy as np
import yaml 
import glob

with open("calibration.yaml", "r") as f:
    loadeddict = yaml.load(f)

mtx = loadeddict.get("camera_matrix")   #focal and optical center
dist = loadeddict.get("dist_coeff")     #distortion

